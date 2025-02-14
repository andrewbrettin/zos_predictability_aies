"""
train.py

This trains the mean network
"""

__author__ = "@andrewbrettin"


#$ Imports
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
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import utils
from utils import settings


#$ Global variables
with open("~/s2s/paths.json") as paths_json:
    PATHS = json.load(paths_json)
with open("~/s2s/globals.json") as globals_json:
    GLOBALS = json.load(globals_json)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(f"Using cuda device {torch.cuda.get_device_name(0)}")

#$ Get configurations passed by batch script
TAU = int(sys.argv[1])
ICOORD = int(sys.argv[2])
MEAN_CONFIGS = settings.configs['mean']
RESIDUAL_CONFIGS = settings.configs['residual']

# Data paths
DATA_PATH = os.path.join(PATHS['full_globe'], 'data', f'tau_{TAU}')
NETWORK_PATH = os.path.join(PATHS['full_globe'], 'networks', f'tau_{TAU}', f'loc_{ICOORD}')

#$ Functions
def get_callbacks(configs, network_type='mean'):
    if network_type == 'mean':
        monitor = 'val/MSE'
    elif network_type == 'residual':
        monitor = 'val/GaussianNLL'
    callback = pl.callbacks.EarlyStopping(
        monitor=monitor, patience=configs['patience'], mode='min')
    return callback

#$ Main
def main_1():
    START_TIME = datetime.now()
    print(f"Begin training mean network {START_TIME}")

    os.makedirs(NETWORK_PATH, exist_ok=True)
    
    #$ Load data        
    X_train = torch.load(os.path.join(DATA_PATH, 'X_train_tensor.pt'))
    y_train = torch.load(os.path.join(DATA_PATH, 'y_train_tensor.pt'))
    X_val = torch.load(os.path.join(DATA_PATH, 'X_val_tensor.pt'))
    y_val = torch.load(os.path.join(DATA_PATH, 'y_val_tensor.pt'))
    
    # Select point for training
    icoords = np.load(os.path.join(DATA_PATH, 'icoords.npy'))
    ilat, ilon = icoords[ICOORD]
    y_train = y_train[:, ilat, ilon].reshape(-1,1)
    y_val = y_val[:, ilat, ilon].reshape(-1,1)
    
    #$ Create torch datasets
    print(f"{datetime.now() - START_TIME}  Creating torch datasets/dataloaders")
    train_dataset = utils.networks.TorchDataset(X_train, y_train)
    val_dataset = utils.networks.TorchDataset(X_val, y_val)
    
    #$ Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=MEAN_CONFIGS['batch_size'],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=MEAN_CONFIGS['batch_size'],
        shuffle=False,
    )
    
    #$ Create ANN
    n_inputs = len(val_dataset[0][0])
    model = utils.networks.ANN(n_inputs, MEAN_CONFIGS)
    
    #$ Create callbacks
    callbacks = get_callbacks(MEAN_CONFIGS, network_type='mean')
    
    #$ Trainer
    accelerator = 'cpu' if DEVICE == 'cpu' else 'gpu'
    trainer = pl.Trainer(
        logger=False,
        accelerator=accelerator,
        max_epochs=MEAN_CONFIGS['epochs'],
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=0,
        num_sanity_val_steps=0,
        default_root_dir=NETWORK_PATH
    )
    
    #$ Train
    print(f"{datetime.now() - START_TIME}  Train network")
    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
    )
    
    #$ Save model
    print(f"{datetime.now() - START_TIME}  Saving model")
    model = model.to('cpu')
    torch.save(model.state_dict(), os.path.join(NETWORK_PATH, 'weights.pth'))
    runtime = datetime.now() - START_TIME
    
    print(f"PROCESS COMPLETED")
    print(datetime.now() - START_TIME)
    return 0

#$ RESIDUAL NETWORK
def main_2():
    START_TIME = datetime.now()
    print(f"Begin training residual network {START_TIME}")
    print(RESIDUAL_CONFIGS)
    
    #$ Load mean model
    mean_model = utils.networks.load_model(
        configs=MEAN_CONFIGS, data_path=DATA_PATH, 
        network_path=NETWORK_PATH, modeltype='mean')
    
    #$ Load data
    X_train = torch.load(os.path.join(DATA_PATH, 'X_train_tensor.pt'))
    y_train = torch.load(os.path.join(DATA_PATH, 'y_train_tensor.pt'))
    X_val = torch.load(os.path.join(DATA_PATH, 'X_val_tensor.pt'))
    y_val = torch.load(os.path.join(DATA_PATH, 'y_val_tensor.pt'))
    
    icoords = np.load(os.path.join(DATA_PATH, 'icoords.npy'))
    ilat, ilon = icoords[ICOORD]
    y_train = y_train[:, ilat, ilon].reshape(-1,1)
    y_val = y_val[:, ilat, ilon].reshape(-1,1)

    #$ Create residuals
    train_residuals = mean_model(X_train).detach() - y_train
    val_residuals = mean_model(X_val).detach() - y_val
    
    #$ Standardize residuals and save residual scaler
    residual_standardizer = {
        'mean': train_residuals.mean().item(),
        'std': train_residuals.std().item()
    }
    
    train_residuals = utils.processing.standardize(
        train_residuals, residual_standardizer)
    val_residuals = utils.processing.standardize(
        val_residuals, residual_standardizer)
    utils.processing.save_standardizer(
        residual_standardizer, data_path=NETWORK_PATH,
        name='residual_standardizer')
    
    #$ Create torch datasets
    print(
        f"{datetime.now() - START_TIME}  Creating residual torch "
        "datasets/dataloaders"
    )
    train_dataset = utils.networks.TorchDataset(X_train, train_residuals)
    val_dataset = utils.networks.TorchDataset(X_val, val_residuals)
    
    #$ Dataloaders
    print(f"{datetime.now() - START_TIME}  Creating dataloaders")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=RESIDUAL_CONFIGS['batch_size'],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=RESIDUAL_CONFIGS['batch_size'],
        shuffle=False,
    )
    
    #$ Create ANN
    n_inputs = len(val_dataset[0][0])
    residual_model = utils.networks.residual_ANN(n_inputs, RESIDUAL_CONFIGS)
    
    #$ Create callbacks
    callbacks = get_callbacks(RESIDUAL_CONFIGS, network_type='residual')

    
    #$ Trainer
    accelerator = 'cpu' if DEVICE == 'cpu' else 'gpu'
    trainer = pl.Trainer(
        logger=False,
        accelerator=accelerator,
        max_epochs=RESIDUAL_CONFIGS['epochs'],
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=0,
        num_sanity_val_steps=0,
        default_root_dir=NETWORK_PATH
    )
    
    #$ Train
    print(f"{datetime.now() - START_TIME}  Train network")
    trainer.fit(
        residual_model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
    )
    
    #$ Save model
    print(f"{datetime.now() - START_TIME}  Saving residual model")
    residual_model = residual_model.to('cpu')
    torch.save(
        residual_model.state_dict(),
        os.path.join(NETWORK_PATH, 'residual_weights.pth')
    )
    
    print(f"PROCESS COMPLETED")
    print(datetime.now() - START_TIME)
    return 0

#$ Run
if __name__ == "__main__":
    main_1()
    main_2()