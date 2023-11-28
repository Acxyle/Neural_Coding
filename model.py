#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:23:20 2023

@author: acxyle-workstation

Note:
    
    Test pytorch version of Github Repository: retinal-crnn_model [link: https://github.com/Zyj061/retina-crnn_model]

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

from torchvision.models import resnet

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
from tqdm import tqdm

import off_data_generator
from torchvision.models import vgg

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
    
class CNNModel(nn.Module):
    
    def __init__(self, bc_size=6, rolling_window=20, cell_num=2, init_weights:bool=True):
        
        super(CNNModel, self).__init__()
        """
            [notice] keras code has an interesting section: kernel_regularizer, seems pytorch can not config this thing
            when build the model but in the training process?
        """
        
        self.conv1 = nn.Conv2d(in_channels=rolling_window, out_channels=8, kernel_size=bc_size, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.noise1 = GaussianNoise(sigma=0.1)  # local
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding='valid')
        self.bn2 = nn.BatchNorm2d(num_features=4)     # [notice] pytorch does not support [single element] BN although keras does
        self.noise2 = GaussianNoise(sigma=0.1)  # local
        
        self.flat = nn.Flatten()
        self.droupout = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(in_features=4, out_features=cell_num)  # dynamic with input shape
        self.bn3 = nn.BatchNorm1d(num_features=cell_num)     # [notice] pytorch does not support [single element] BN although keras does
        #self.parametric_softplus = ParametricSoftplus(size=(2,))  # local
        self.softplus = nn.Softplus()        
        
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
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.noise2(x)
        x = F.relu(x)

        x = self.flat(x)
        x = self.droupout(x)
        
        x = self.fc1(x)
        x = self.bn3(x)
        #x = self.parametric_softplus(x)
        x = self.softplus(x)
        
        return x


data_prefix = './data/'
bc_size=6


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

def train_model(BATCHSIZE=4096, VAL_BATCHSIZE=2048, rolling_window=20, num_epochs=1000):
    
    # -----   
    device = 'cuda'
    # -----
    
    l1_reg = 1e-3
    
    log_path = './off_cnn_bc' + str(bc_size) + '_log'
    
    model = CNNModel(bc_size=bc_size).to(device)
    
    model.train()

    # 1. Data Preparation
    train_dataset = off_data_generator.ImageGenerator_pytorch('train', rolling_window)
    val_dataset = off_data_generator.ImageGenerator_pytorch('test', rolling_window)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, num_workers=24, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=VAL_BATCHSIZE, num_workers=24, shuffle=False)

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

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            
            l1_loss = l1_regularizer(model, 'conv1', l1_reg)
            loss += l1_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_metric += cc_pytorch(outputs, labels).item()
            
            if i == math.floor(len(train_dataset)/BATCHSIZE):

                val_loss = 0.
                val_metric = 0.
                
                model.eval()    
                
                with torch.no_grad():
                    for j, (val_inputs, val_labels) in enumerate(val_loader, 0):
                        
                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        val_outputs = model(val_inputs)
                        
                        val_loss += criterion(val_outputs, val_labels).item()
                        val_metric += cc_pytorch(val_outputs, val_labels).item()

                val_loss /= len(val_loader)
                val_metric /= len(val_loader)
                
                print(f'Epoch {epoch + 1}, Loss: {running_loss/len(train_loader):.5f}, CC: {running_metric/len(train_loader):.5f}, Val Loss: {val_loss:.5f}, Val CC: {val_metric:.5f}')
                
                model.train()
                
        # Check early stopping condition
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
            if patience_counter > patience:
                print('Early stopping')
                return model
        
        print(f'Best_loss: {best_loss:.5f}, Patience: {patience_counter:.5f}')
                
                

    return model



if __name__ == "__main__":
    
    train_model()
    
    #model = CNNModel()
    #x = torch.full((2,20,8,8),2.)
    #y = model(x)
    