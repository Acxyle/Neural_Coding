import os, random
import h5py
import scipy.io as scio
import numpy as np
from collections import namedtuple
from scipy.ndimage.filters import gaussian_filter1d

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

import utils_


data_prefix = './data'

filename = 'cell_simpleNL_off_2GC_v3.mat'

filepath = os.path.join(data_prefix, filename)

Exptdata = namedtuple('Exptdata', ['X', 'r'])

def rolling_window(arr, history, time_axis=-1):
    """
        this function creates a 'view' of the data - arr with a sliding window with size - history
        eg: 
            original data [1,2,3,4,5]
            reviewed data [[1,2,3], [2,3,4], [3,4,5]]
    """

    if time_axis == 0:
        arr = arr.T
    elif time_axis == -1:
        pass
    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')
    
    assert history >= 1, "'window' must be least 1."
    assert history < arr.shape[-1], "'window' is longer than array"

    #with strides 
    shape = arr.shape[:-1] + (arr.shape[-1] - history + 1, history)
    strides = arr.strides + (arr.strides[-1], )
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr, 1, 0)
    else:
        return arr


def load_data(phase, history):
    """
        this function loads data from data directory and 
            1. save them into 2 .h5 datasets
            2. return wrapped 'Exptdata' type namedtuple
    """

    train_filepath = os.path.join(data_prefix, 'train_off3.hdf5')
    test_filepath = os.path.join(data_prefix, 'test_off3.hdf5')
    
    # ----- if the data has not been extracted from the .mat file
    if not os.path.exists(train_filepath):
        data = scio.loadmat(filepath)

        # loading different type of data
        stim = data['CB']     # (8, 8, 300000)

        # split data into train set and validation set
        stim_size = 3e5
        trn_stim_size = int(stim_size / 6 * 5)
        trn_x = stim[:, :, :trn_stim_size]     # (8, 8, 250000) 5/6 as training set
        tst_x = stim[:, :, trn_stim_size:]     # (8, 8, 50000) 1/6 as val set
        
        r = data['fr']
        r = r.T     # (2, 300000)

        trn_r = r[:, :trn_stim_size]     # (2, 250000) 5/6
        tst_r = r[:, trn_stim_size:]     # (2, 50000) 1/6

        with h5py.File(train_filepath, 'w') as fw:
            fw.create_dataset('trn_x', data=trn_x)
            fw.create_dataset('trn_r', data=trn_r)

        with h5py.File(test_filepath, 'w') as fw:
            fw.create_dataset('tst_x', data=tst_x)
            fw.create_dataset('tst_r', data=tst_r)
        
    train_data = h5py.File(train_filepath, 'r')
    test_data = h5py.File(test_filepath, 'r')
    
    X = train_data['trn_x']
    r = train_data['trn_r']

    if history > 1:
        X = rolling_window(np.array(X), history)     # 'view' the data
        X = np.transpose(X, (0, 1, 3, 2)) 
        r = r[:, history-1:]
    else:
        X = np.array(X)
        r = np.array(r)
        X = X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])
    
    r = r.T

    trn_data = Exptdata(X, r)     # use named tuple to encapsulate the data - data and label pair

    tst_X = test_data['tst_x']
    tst_r = test_data['tst_r']

    if history > 1:
        tst_X = rolling_window(np.array(tst_X), history)
        tst_X = np.transpose(tst_X, (0, 1, 3, 2))
        tst_r = tst_r[:, history-1:]
    else:
        tst_X = np.array(tst_X)
        tst_r = np.array(tst_r)
        tst_X = tst_X.reshape(tst_X.shape[0], tst_X.shape[1], 1, tst_X.shape[2])

    tst_r = tst_r.T
    tst_data = Exptdata(tst_X, tst_r)

    if phase == 'train':
        return trn_data
    else:
        return tst_data


# ----------------------------------------------------------------------------------------------------------------------
class ImageGenerator_pytorch(Dataset):
    
    def __init__(self, phase, history):

        self.img_h = 8
        self.img_w = 8
        self.cell_num = 2  # 2 RGC off cells to fit
        
        self.phase = phase
        self.history = history
        
        self.data = load_data(self.phase, self.history)
        
        self.num_data = self.data.X.shape[-1]
        self.indexes = list(range(self.num_data))

    def __len__(self):

        return self.num_data

    def __getitem__(self, idx):

        x = self.data.X[:, :, :, idx]
        y = self.data.r[idx, :]

        data = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)  # PyTorch uses (C, H, W) format
        label = torch.tensor(y, dtype=torch.float32)

        return data, label



if __name__ == "__main__":
    
    train_dataset = ImageGenerator_pytorch('train', 20)
    
    # Create an instance of the DataLoader
    pytorch_loader = DataLoader(train_dataset, batch_size=4096, shuffle=False)
    
    # Iterate through the DataLoader to get the first batch
    for batch in pytorch_loader:
        X, Y = batch
        print("Features (X):", X)
        print("Labels (Y):", Y)
        break  # Exit the loop after the first batch
