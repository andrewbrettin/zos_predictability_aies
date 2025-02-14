"""
postprocess.py

This computes the predicted mean and standard devations on the test dataset and saves them.
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

sys.path.append('~/zos_predictability/training/')
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
PREDICTIONS_PATH = os.path.join(PATHS['full_globe'], 'predictions', f'tau_{TAU}', f'loc_{ICOORD}')

def load_tensors(data_path, icoord_index):
    # Load all icoords
    icoords = np.load(os.path.join(DATA_PATH, 'icoords.npy'))
    ilat, ilon = icoords[icoord_index]
    
    # Get data
    X_test_tensor = torch.load(os.path.join(data_path, 'X_test_tensor.pt'))
    y_test_tensor = torch.load(os.path.join(data_path, 'y_test_tensor.pt'))
    y_test_tensor = y_test_tensor[:, ilat, ilon]
    
    return X_test_tensor, y_test_tensor

def get_models(network_path, n_inputs, mean_configs, residual_configs):
    # Get mean and residual model
    mean_model = utils.networks.ANN(n_inputs, mean_configs)
    weights = torch.load(os.path.join(network_path, 'weights.pth'))
    mean_model.load_state_dict(weights)
    mean_model.eval()

    residual_model = utils.networks.residual_ANN(n_inputs, residual_configs)
    weights = torch.load(os.path.join(network_path, 'residual_weights.pth'))
    residual_model.load_state_dict(weights)
    residual_model.eval()

    return mean_model, residual_model

def get_predictions(mean_model, residual_model, X_test_tensor):
    pred_mean_z = mean_model(X_test_tensor).detach().numpy().flatten()
    pred_logvar_z = residual_model(X_test_tensor).detach().numpy().flatten()
    
    return pred_mean_z, pred_logvar_z
    

def main():
    # Make paths
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    
    # Get data
    X_test_tensor, y_test_tensor = load_tensors(DATA_PATH, ICOORD)
    n_inputs = X_test_tensor.shape[1]
    
    # Get models
    mean_model, residual_model = get_models(NETWORK_PATH, n_inputs, MEAN_CONFIGS, RESIDUAL_CONFIGS)
    
    # Predictions
    pred_mean_z, pred_logvar_z = get_predictions(mean_model, residual_model, X_test_tensor)
    
    # Save predictions
    mean_pred_filename = os.path.join(PREDICTIONS_PATH, 'pred_mean_z.npy')
    logvar_pred_filename = os.path.join(PREDICTIONS_PATH, 'pred_logvar_z.npy')
    np.save(mean_pred_filename, pred_mean_z)
    np.save(logvar_pred_filename, pred_logvar_z)    
    
    print("Postprocessing completed")
    return 0


if __name__ == "__main__":
    main()
