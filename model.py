#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:23:20 2023

@author: acxyle-workstation

Note:
    
    Test spikingjelly pytorch version of Github Repository: retinal-crnn_model [link: https://github.com/Zyj061/retina-crnn_model]

Task:
    
    1. pytorch replication of the Retina Neural Coding
    
        1.1 Rebuild the network:
            1.1.1. Dataset
            1.1.2. Dataloader
            1.1.3. ParametricSoftplus
            1.1.4. GaussianNoise
            1.1.5. L1Loss
        -----
        Equivalent Check: pending...
        -----
        
        1.2 Rebuild the training process
            1.2.1 
            
        -----
        Equivalent Check: pending...
        -----
            
        1.3 Rebuild/Link the data analysis
            1.3.1
            
        -----
        Equivalent Check: pending...
        -----
    
    2. SNN fir Retina Neural Coding
    
"""

import os
import h5py
from scipy.stats import pearsonr
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spikingjelly.activation_based import functional, neuron, layer, surrogate

import off_data_generator
from torchvision.models import vgg

from tv_ref_classify import utils

import visualizations


import utils_

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.sigma
        return x

class ParametricSoftplus(nn.Module):
    def __init__(self, alpha_init=0.2, beta_init=5.0, size=None):
       super(ParametricSoftplus, self).__init__()

       self.size = size
       
       if size is None:
           self.alpha = nn.Parameter(torch.full((1,), alpha_init))
           self.beta = nn.Parameter(torch.full((1,), beta_init))
       else:
           self.alpha = nn.Parameter(torch.full(size, alpha_init))
           self.beta = nn.Parameter(torch.full(size, beta_init))

    def forward(self, x):

        return F.softplus(self.beta * x) * self.alpha
    
class SpikingCNNModel(nn.Module):
    
    def __init__(self, bc_size=6, rolling_window=20, cell_num=2, init_weights:bool=True, 
                 spiking_neuron:callable=None, **kwargs):
        
        super(SpikingCNNModel, self).__init__()

        self.conv1 = layer.Conv2d(in_channels=rolling_window, out_channels=8, kernel_size=bc_size, padding=0)
        self.bn1 = layer.BatchNorm2d(8)
        self.noise1 = GaussianNoise(sigma=0.1)  # local
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv2 = layer.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding='valid')
        self.bn2 = layer.BatchNorm2d(4)
        self.noise2 = GaussianNoise(sigma=0.1)  # local
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        
        self.droupout = layer.Dropout(p=0.2)
        
        self.fc1 = layer.Linear(in_features=4, out_features=cell_num)  # dynamic with input shape
        self.bn3 = layer.BatchNorm1d(cell_num)
        #self.parametric_softplus = ParametricSoftplus(size=(2,))  # local        
        
        # ---
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.noise1(x)
        
        x = self.sn1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.noise2(x)
        
        x = self.sn2(x)
        
        if self.sn2.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.sn2.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.droupout(x)
        
        x = self.fc1(x)
        x = self.bn3(x.unsqueeze(3)).squeeze()
        
        #x = self.parametric_softplus(x)
        x = nn.Softplus()(x)
        
        return x


data_prefix = './data/'
bc_size=6


# computing the pearson correlation for multi cell under array mode
def cc_np(x, y):
    """
        input:
            x: bio cells' performances
            y: NN predictions
    """

    num_cell = x.shape[1]
    cc_res = np.zeros((num_cell, ))
    
    for i in range(num_cell):
        cc_res[i] = pearsonr(x[:, i], y[:, i])[0]

    return np.mean(cc_res)


def cc_pytorch(x, y):
    """
        Metric: Pearson's correlation coefficient
    """
    
    def _ss(a):
        return torch.bmm(a.unsqueeze(1), a.unsqueeze(2)).squeeze()
    
    mx = torch.mean(x, 0, keepdim=True)
    my = torch.mean(y, 0, keepdim=True)

    xm, ym = x - mx, y - my
    
    r_num = torch.bmm(xm.unsqueeze(1), ym.unsqueeze(2)).squeeze()
    r_den = torch.sqrt(_ss(xm) * _ss(ym) + 1e-8)

    r = torch.mean(r_num/r_den)
    
    return r

def l1_regularizer(model, layer_name, lambda_l1):

    return lambda_l1 * sum(p[1].abs().sum() for p in model.named_parameters() if layer_name in p[0])


def preprocess_train_sample(T, x:torch.Tensor):
        return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

def preprocess_test_sample(T, x:torch.Tensor):
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

def process_model_output(y: torch.Tensor):
    return y.mean(0)  # return firing rate

import argparse

parser = argparse.ArgumentParser(description="SpikingCNN test")
parser.add_argument("--device", default="cuda", type=str, help="model name")
        
args = parser.parse_args()

def train_model(BATCHSIZE=4096, VAL_BATCHSIZE=2048, rolling_window=20, num_epochs=1000, 
                T=4, neuron_='IF'):
    
    if neuron_ == 'IF':
        spiking_neuron = neuron.IFNode
    elif neuron_ == 'LIF':
        spiking_neuron = neuron.LIFNode
    else:
        raise ValueError('[Codinfo] Invalid neuron {neuron}')
    
    
    utils.init_distributed_mode(args)
    
    # -----   
    device = 'cuda'
    # -----
    
    l1_reg = 1e-3
    
    log_path = './off_cnn_bc' + str(bc_size) + '_log'
    
    model = SpikingCNNModel(bc_size=bc_size, spiking_neuron=spiking_neuron, surrogate_function=surrogate.ATan(), detach_reset=True).to(device)
    functional.set_step_mode(model, step_mode='m')
    
    model.train()

    # 1. Data Preparation
    train_dataset = off_data_generator.ImageGenerator_pytorch('train', rolling_window)
    val_dataset = off_data_generator.ImageGenerator_pytorch('test', rolling_window)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, num_workers=16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=VAL_BATCHSIZE, num_workers=16, shuffle=False)

    # 2. Loss function and optimizer
    criterion = nn.PoissonNLLLoss(log_input=False) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # TensorBoard writer
    #writer = SummaryWriter(log_path)

    # Early stopping parameters
    min_delta = 0.0001
    patience = 50
    patience_counter = 0
    
    best_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.
        running_metric = 0.
        
        for i, (inputs, labels) in tqdm(enumerate(train_loader, 0), desc=f'Epoch {epoch}'):
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            inputs = preprocess_train_sample(T, inputs)

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = process_model_output(outputs)

            loss = criterion(outputs, labels)
            
            l1_loss = l1_regularizer(model, 'conv1', l1_reg)
            loss += l1_loss

            loss.backward()
            optimizer.step()
            functional.reset_net(model)

            running_loss += loss.item()
            running_metric += cc_pytorch(outputs, labels).item()
            
            if i == math.floor(len(train_dataset)/BATCHSIZE):     # for the last batch, i.e. the end of the epoch

                val_loss = 0.
                val_metric = 0.
                
                model.eval()    
                
                with torch.inference_mode():
                    for j, (val_inputs, val_labels) in enumerate(val_loader, 0):
                        
                        val_inputs, val_labels = val_inputs.to(device, non_blocking=True), val_labels.to(device, non_blocking=True)
                        val_inputs = preprocess_test_sample(T, val_inputs)
                        
                        val_outputs = model(val_inputs)
                        val_outputs = process_model_output(val_outputs)
                        
                        val_loss += criterion(val_outputs, val_labels).item()
                        val_metric += cc_pytorch(val_outputs, val_labels).item()

                        functional.reset_net(model)
                
                val_loss /= len(val_loader)
                val_metric /= len(val_loader)
                
                print(f'Epoch {epoch + 1}, Loss: {running_loss/len(train_loader):.5f}, CC: {running_metric/len(train_loader):.5f}, Val Loss: {val_loss:.5f}, Val CC: {val_metric:.5f}')
                
                model.train()
                
                # Check early stopping condition
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    patience_counter = 0
                    
                    torch.save(model.state_dict(), f'./offline_model_weights_spiking_cnn_{neuron_}_T{T}.pth')
                    
                else:
                    patience_counter += 1
                    
                    if patience_counter > patience:
                        print('Early stopping')
                        return model
        
                print(f'Best_loss: {best_loss:.5f}, Patience: {patience_counter}')
                
        
def test_model_pytorch(bc_size=6, device='cpu', output_path=None, history=20, neuron_='IF', T=4):
    
    if output_path == None:
        output_path = f'{neuron_}_T{T}'
    
    if neuron_ == 'IF':
        spiking_neuron = neuron.IFNode
    elif neuron_ == 'LIF':
        spiking_neuron = neuron.LIFNode
    else:
        raise ValueError('[Codinfo] Invalid neuron {neuron}')
    
    # --- 1. recover model
    model = SpikingCNNModel(bc_size=bc_size, spiking_neuron=spiking_neuron, surrogate_function=surrogate.ATan(), detach_reset=True).to(device)
    functional.set_step_mode(model, step_mode='m')
    
    weights_path = f'./offline_model_weights_spiking_cnn_{neuron_}_T{T}.pth'
    parameters = torch.load(weights_path)
    
    model.load_state_dict(parameters)

    model.eval()  
    
    print('6')
    
    val_data = off_data_generator.load_data('test', history)
    
    # --- 2. visualization for prediction
    vis_fr_pytorch(model, val_data, output_path, neuron_, T)
    
    
    # --- 3. kernel analysis
    tmp = [_ for _ in model.named_parameters()]
    tmp = {_[0]:_[1] for _ in tmp}
    conv1_k = tmp['conv1.weight'].detach().numpy()
    
    fig = visualizations.plot_filters(conv1_k)
    fig.suptitle(f'SpikingCNN (spikingjelly) {neuron_} {T} Conv1 8 kernels')
    fig.savefig(os.path.join(output_path, 'conv1_kernel.png'))
    fig.savefig(os.path.join(output_path, 'conv1_kernel.eps'))

    

def vis_fr_pytorch(model, data, output_path='', neuron_=None, T=4, display=True, device='cpu'):
    
    X = torch.tensor(data.X, dtype=torch.float32).to(device)
    y = torch.tensor(data.r, dtype=torch.float32).to(device) 
    
    X = X.permute(3, 2, 0, 1) 
    
    X = preprocess_test_sample(T, X)
    y_pred = model(X)  
    y_pred = process_model_output(y_pred)
    
    timewindow = y_pred.shape[0]
    num_cell = y_pred.shape[1]
    
    response_filepath = os.path.join(output_path, 'response_test')
    utils_.make_dir(response_filepath)
    
    # -----
    y_np = y.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    cc_res = cc_np(y_np, y_pred_np)
    print('cc: ' + str(cc_res))

    if display is True:
        with h5py.File(response_filepath + '/result.hdf5', 'w') as fw:
            fw.create_dataset('r_data', data=y_np)
            fw.create_dataset('r_pre', data=y_pred_np)

        for i in range(num_cell):
            y_max = y_np[:, i].max()
            y_min = y_np[:, i].min() 
            y_pre_max = y_pred_np[:, i].max()
            y_pre_min = y_pred_np[:, i].min()

            if (y_max - y_min) == 0:
                break
        
        # ---
        for i in range(num_cell):

            figpath = response_filepath + '/' + str(i)
            plot_fr(y_np[:, i].T, y_pred_np[:, i].T, figpath, i, neuron_, T)
        
    # Compute average firing rate
    y_avg = torch.mean(y, axis=1)
    y_pre_avg = torch.mean(y_pred, axis=1)

    if display is True:
        figpath = response_filepath + '/avg'
        avg_cc = plot_fr(y_avg.numpy(), y_pre_avg.detach().numpy(), figpath, neuron_, T)
    else:
        avg_cc = pearsonr(y_avg.numpy(), y_pre_avg.detach().numpy())[0]

    print('cc on population cell: %f' % avg_cc)

    return cc_res, avg_cc

import matplotlib.pyplot as plt
# plot the firing rate 
def plot_fr(r, r_pre, save_path, cell_id=None, neuron_=None, T=None):

    timewindow = len(r)

    ig, ax = plt.subplots(figsize=(20,6))
    ax.plot(np.arange(len(r)), r, linewidth=1, color='blue', alpha=0.5, label='data')
    ax.plot(np.arange(len(r_pre)), r_pre, linewidth=1, color='red', alpha=0.5, label='prediction')
    ax.set_ylabel('Rate', fontsize=18)
    ax.set_xlim([0, timewindow])
    ax.legend(loc='upper left')

    cc_res = pearsonr(r, r_pre)[0]
    
    ax.set_title(f'SpikingCNN [{neuron_}] [{T}] Cell [{cell_id}] with cc [{cc_res:.5f}]', fontsize=18)
    
    plt.tight_layout()
    plt.savefig(save_path+'.png')
    plt.savefig(save_path+'.eps')
    plt.close()
    
    return cc_res

if __name__ == "__main__":
    
    #model = SpikingCNNModel(spiking_neuron=neuron.IFNode)
    #x = torch.full((2,20,8,8),2.)
    #y = model(x)
    
    # -----
    train_model(T=200, neuron_='LIF')
    
    #test_model_pytorch(T=4, neuron_='LIF')
    