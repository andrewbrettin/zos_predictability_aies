"""
networks.py

Various utility functions for dealing with networks.
"""


__author__ = "@andrewbrettin"
__all__ = [
    "XarrayTorchDataset",
    "TorchDataset",
    "DampedPersistenceLoss",
    "ANN",
    "residual_ANN",
    "load_model",
    "get_predictions"
]

from typing import Any, Callable, Dict, Optional, Tuple
import os
import sys
import json
from datetime import datetime

import random
import numpy as np
import scipy as sc
import pandas as pd
import xarray as xr
import cftime

from dask.diagnostics import ProgressBar
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
import wandb

from . import processing, data
sys.path.append('..')

with open("~/s2s/paths.json") as paths_json:
    PATHS = json.load(paths_json)
with open("~/s2s/globals.json") as globals_json:
    GLOBALS = json.load(globals_json)


#$ Classes
class XarrayTorchDataset(torch.utils.data.Dataset):
    def __init__(self, X_ds, y_ds):
        super(XarrayTorchDataset, self).__init__()
        self.X_tensor = X_ds.values
        self.y_tensor = y_ds.values
        
        self.X_tensor = torch.from_numpy(self.X_tensor)
        self.y_tensor = torch.from_numpy(self.y_tensor).reshape(-1,1)
        
        # Keep coordinate info just in case
        self.X_coords = X_ds.coords
        self.y_coords = y_ds.coords
        
    def __len__(self):
        return len(self.y_tensor)
    
    def __getitem__(self, idx):
        return self.X_tensor[idx, :], self.y_tensor[idx]

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X_tensor, y_tensor):
        super(TorchDataset, self).__init__()
        self.X_tensor = X_tensor
        self.y_tensor = y_tensor
    
    def __len__(self):
        return len(self.y_tensor)
    
    def __getitem__(self, idx):
        return self.X_tensor[idx, :], self.y_tensor[idx]
        

class DampedPersistenceLoss(nn.Module):
    def __init__(self, beta):
        super(DampedPersistenceLoss, self).__init__()
        self.beta = beta
        
    def forward(self, input, target):
        return torch.abs(target - self.beta * input)
    
class ANN(pl.LightningModule):
    def __init__(self, n_inputs, configs):
        super(ANN, self).__init__()
        
        #$ Organize hidden layers
        hiddens_list = configs['network_architecture']
        hidden_layers = []
        for i, n in enumerate(hiddens_list[0:-1]):
            np1 = hiddens_list[i+1]
            hidden_layers.append(nn.Linear(n, np1))
            hidden_layers.append(nn.ReLU())
        hidden_layers = nn.ModuleList(hidden_layers)
        
        #$ Create stack (with or without dropout layers
        if configs['dropout'] is None:
            self.stack = nn.Sequential(
                nn.Linear(n_inputs, hiddens_list[0]),      # Input layer
                nn.ReLU(),
                *hidden_layers,                            # Hidden layers
                nn.Linear(hiddens_list[-1], 1)             # Output layer
            )
        else:
            assert isinstance(configs['dropout'], float)
            self.stack = nn.Sequential(
                nn.Linear(n_inputs, hiddens_list[0]),      # Input layer
                nn.Dropout(p=configs['dropout']),
                nn.ReLU(),
                *hidden_layers,                            # Hidden layers
                nn.Linear(hiddens_list[-1], 1)             # Output layer
            )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        #$ Hyperparameter configs
        self.configs = configs
        
    def forward(self, x):
        return self.stack(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self.stack(X)
        loss = self.loss_fn(y, pred)
        
        self.log('train/MSE', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self.stack(X)
        loss = self.loss_fn(y, pred)
        
        self.log('val/MSE', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer_func = getattr(torch.optim, self.configs['optimizer'])
        optimizer = optimizer_func(
            self.stack.parameters(),
            lr=self.configs['lr'],
            weight_decay=self.configs['l2']
        )
        return optimizer


class residual_ANN(pl.LightningModule):
    def __init__(self, n_inputs, configs):
        super(residual_ANN, self).__init__()
        
        #$ Organize hidden layers
        hiddens_list = configs['network_architecture']
        hidden_layers = []
        for i, n in enumerate(hiddens_list[0:-1]):
            np1 = hiddens_list[i+1]
            hidden_layers.append(nn.Linear(n, np1))
            hidden_layers.append(nn.ReLU())
        hidden_layers = nn.ModuleList(hidden_layers)
        
        #$ Create stack (with or without dropout layers)
        if configs['dropout'] is None:
            self.stack = nn.Sequential(
                nn.Linear(n_inputs, hiddens_list[0]),      # Input layer
                nn.ReLU(),
                *hidden_layers,                            # Hidden layers
                nn.Linear(hiddens_list[-1], 1)             # Output layer
            )
        else:
            assert isinstance(configs['dropout'], float)
            self.stack = nn.Sequential(
                nn.Linear(n_inputs, hiddens_list[0]),      # Input layer
                nn.Dropout(p=configs['dropout']),
                nn.ReLU(),
                *hidden_layers,                            # Hidden layers
                nn.Linear(hiddens_list[-1], 1)             # Output layer
            )
        
        #$ Hyperparameter configs
        self.configs = configs
        
        #$ Loss function
        self.loss_fn = nn.GaussianNLLLoss()
        
    def forward(self, x):
        return self.stack(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self.stack(X)
        loss = self.loss_fn(y, 0, torch.exp(pred))
        
        self.log('train/GaussianNLL', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self.stack(X)
        loss = self.loss_fn(y, 0, torch.exp(pred))
        
        self.log('val/GaussianNLL', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer_func = getattr(torch.optim, self.configs['optimizer'])
        optimizer = optimizer_func(
            self.stack.parameters(),
            lr=self.configs['lr'],
            weight_decay=self.configs['l2']
        )
        return optimizer

def load_model(configs, data_path, network_path, modeltype='mean'):
    X_test = torch.load(os.path.join(data_path, 'X_test_tensor.pt'))
    n_samples, n_inputs = X_test.shape
    
    if modeltype == 'mean':
        model = ANN(n_inputs, configs)
        weights = torch.load(os.path.join(network_path, 'weights.pth'))
        model.load_state_dict(weights)
    else:
        model = residual_ANN(n_inputs, configs)
        weights = torch.load(os.path.join(network_path, 'residual_weights.pth'))
        model.load_state_dict(weights)
    
    return model


def get_predictions(configs, data_path, network_path, sampling=5, standardized=False):
    mean_model = load_model(
        configs=configs,
        data_path=data_path, 
        network_path=network_path
    )
    
    residual_model = load_model(
        configs=configs, 
        data_path=data_path,
        network_path=network_path,
        modeltype='residual'
    )
    
    # Get data and standardizers
    # Standardizers
    X_scaler = processing.load_standardizer(data_path, 'X')
    y_scaler = processing.load_standardizer(data_path, 'y')
    residual_scaler = processing.load_standardizer(network_path, 'residual')

    # Data
    X_test = data.load_flattened_features(data_path, 'test')
    y_test = data.load_flattened_targets(data_path, 'test')

    # Torch Dataset
    X_test_tensor = torch.from_numpy(X_test.values).float()
    y_test_tensor = torch.from_numpy(y_test.values).float().reshape(-1,1)
    
    # Target values
    target_z = (
        y_test_tensor
        .detach()
        .numpy()
        .reshape(-1)
    )
    target = processing.unstandardize(target_z, y_scaler)
    
    # Predicted point estimates
    # Standardized predictions
    pred_mean_z = (
        mean_model(X_test_tensor)
        .detach()
        .numpy()
        .flatten()
    )
    # Unstandardized mean
    pred_mean = processing.unstandardize(pred_mean_z, y_scaler)
    
    # Predicted uncertainties
    # 1. Compute prediction and standardized sigma
    pred_std_zz = (
        residual_model(X_test_tensor)
        .exp()
        .sqrt()
        .detach()
        .numpy()
        .flatten()
    )
    # 2. Standardize according to residuals
    pred_std_z = processing.unstandardize(pred_std_zz, residual_scaler)
    # 3. Scale by mean standardizer sigma
    pred_std = y_scaler['std'] * pred_std_z
    
    # Make sure xarray cftimes are ok
    if isinstance(y_test['time'][0].item(), str):
        cftimes = xr.cftime_range(
            y_test['time'][0].item(),
            y_test['time'][-1].item(),
            freq=f'{sampling}D',
            calendar='noleap'
        )
    else:
        cftimes = y_test['time']

    cftimes = xr.DataArray(cftimes, coords={'time': cftimes})
    
    # Define DataArrays
    pred_mean = xr.DataArray(pred_mean, coords={'time': cftimes}, name='mu')
    pred_std = xr.DataArray(pred_std, coords={'time': cftimes}, name='sigma')
    target = xr.DataArray(target, coords={'time': cftimes}, name='true')
    
    if standardized:
        return pred_mean_z, pred_std_z, target_z
    else:
        return pred_mean, pred_std, target
    
