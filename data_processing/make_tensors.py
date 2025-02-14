"""
make_tensors.py

"""

__author__ = "@andrewbrettin"

import os
import sys
import json
from datetime import datetime
from itertools import product
import numpy as np
import scipy as sc
import pandas as pd
import xarray as xr
import cftime
from dask.diagnostics import ProgressBar
from dask_jobqueue import PBSCluster
from dask.distributed import Client
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

import utils

#$ Project globals
with open("~/s2s/paths.json") as paths_json: 
    PATHS = json.load(paths_json)
with open("~/s2s/globals.json") as globals_json:
    GLOBALS = json.load(globals_json)

#$ Globals
INPUT_VARIABLES = ['zos', 'SST', 'UBOT', 'VBOT']
T = 5    # Predict 5-day averages
RES = 5  # Input resolution
REGION = 'global_no_poles'
TAU = int(sys.argv[1])

#$ Paths
DATA_PATH = os.path.join(PATHS['full_globe'], 'data', f'tau_{TAU}')


#$ Functions
def create_datasets(tau, T, res, region):
    X_data = []
    y_data = []
    all_members = list(product(GLOBALS['init_years'], GLOBALS['members']))
    
    for i, (init_year, member) in enumerate(all_members):
        #$ Make targets
        #$$ Uncoarsened dynamic sea level anomalies
        zos = utils.data.load_anomalies('zos', init_year, member, chunkedby='space')
        
        # Select points between (-60, 60) only
        zos = zos.sel(lat=slice(-60,60))

        #$$ Compute rolling means (subsample later)
        y = (
            zos
            .rolling(time=T, center=False).mean()  # Rolling mean faster than resample
            .isel(time=slice(T-1, None, T))        # isel faster than dropna; resample
            .shift(time=-tau // T)                 # Did slicing  
            .isel(time=slice(0, -tau // T))        # isel faster than dropna
        )

        #$ Get features
        #$$ If there is only one input variable, load it at the
        #$$ specified resolution. Otherwise, load 4D: var/lat/lon/time
        if len(INPUT_VARIABLES) == 1:
            var = INPUT_VARIABLES[0]
            X = utils.data.load_coarsened_anomalies(var, res, init_year, member)
        else:
            X = xr.Dataset({
                var: utils.data.load_coarsened_anomalies(var, res, init_year, member)
                for var in INPUT_VARIABLES
            })
            X = X.to_array(dim='var')

        #$ Select region
        if region is not None:
            X = utils.data.select_region(X, region)

        #$ For array_job_x5_y5, we should also do 5-day rolling means of quantities
        X = X.rolling(time=T, center=False).mean()
        
        #$$ Use same samples as targets
        time = y.time
        X = X.sel(time=y.time.values)

        #$ Rename
        X.name = f"{X.name}_{i}"
        y.name = f"{y.name}_{i}"

        X.attrs['init_year'] = init_year
        X.attrs['member'] = member
        y.attrs['init_year'] = init_year
        y.attrs['member'] = member

        #$ Append variables
        X_data.append(X)
        y_data.append(y)
        
    #$ Merge variables
    X_all = xr.merge(X_data)
    X_all = X_all.to_array(dim='member_id')
    X_all['member_id'] = np.arange(len(X_data))
    y_all = xr.merge(y_data)
    y_all = y_all.to_array(dim='member_id')
    y_all['member_id'] = np.arange(len(y_data))

    #$ Create sample dimension to reduce dimensionality
    X_all = X_all.stack(sample=('member_id', 'time'))
    y_all = y_all.stack(sample=('member_id', 'time'))
    
    return X_all, y_all

def stack_s(X):
    """
    Flattens X with dims ([var,] lat, lon, sample) into dataarray with
    dims (sample, s) (where s=([var,] lat, lon)).
    """
    
    if 'time' in X.dims:
        X = X.rename({'time': 'sample'})
    if len(INPUT_VARIABLES) == 1:
        spatial_multiindex = (
            X.isel(sample=0)
            .drop('sample')
            .stack(s=('lat','lon'))
            .dropna(dim='s')
            .s
        )
        X = (
            X.stack(s=('lat','lon'))
            .sel(s=spatial_multiindex)
        )
    else:
        spatial_multiindex = (
            X.isel(sample=0)
            .drop('sample')
            .stack(s=('lat','lon','var'))
            .dropna(dim='s')
            .s
        )
        X = (
            X.stack(s=('lat','lon','var'))
            .sel(s=spatial_multiindex)
        )
    return X

def get_train_test_val(ds_all):
    val_mask = ds_all['member_id'] == 5
    val_ds = ds_all.sel(sample=val_mask)
    
    test_mask = ds_all['member_id'] == 8
    test_ds = ds_all.sel(sample=test_mask)
    
    # All other indices are train indices
    train_idxs = [0, 1, 2, 3, 4, 6, 7]
    train_mask = np.isin(ds_all['member_id'], train_idxs)
    train_ds = ds_all.sel(sample=train_mask)
    return train_ds, val_ds, test_ds


def get_train_members(val_member=(1281, '013'), test_member = (1301, '013')):
    all_members = list(product(GLOBALS['init_years'], GLOBALS['members']))
    train_members = all_members.copy()
    train_members.remove(val_member)
    train_members.remove(test_member)
    return train_members

# -------------------------------------------------------------------------- #
def main():
    START_TIME = datetime.now()
    print(f"File:\t {__file__}")
    print(f"Date:\t {START_TIME}")

    print(f"Output path: {DATA_PATH}")
    print("Beginning simulation")
    print()

    # Make full datasets
    print(f"{datetime.now() - START_TIME}\t Creating full datasets")
    with ProgressBar():
        X_all, y_all = create_datasets(TAU, T, RES, REGION)

    # Split into train/val/test
    print(f"{datetime.now() - START_TIME}\t Splitting into train/val/test")
    with ProgressBar():
        X_train, X_val, X_test = get_train_test_val(X_all)
        y_train, y_val, y_test = get_train_test_val(y_all)

    # Create standardizer
    print(f"{datetime.now() - START_TIME}\t Creating standardizers")
    with ProgressBar():
        X_standardizer = utils.processing.get_standardizer(X_train, dim='sample')
        y_standardizer = utils.processing.get_standardizer(y_train, dim='sample')

    # Standardizing
    print(f"{datetime.now() - START_TIME}\t Standardizing")
    with ProgressBar():
        X_train = utils.processing.standardize(X_train, X_standardizer)
        X_val = utils.processing.standardize(X_val, X_standardizer)
        X_test = utils.processing.standardize(X_test, X_standardizer)
        y_train = utils.processing.standardize(y_train, y_standardizer)
        y_val = utils.processing.standardize(y_val, y_standardizer)
        y_test = utils.processing.standardize(y_test, y_standardizer)

    # Flatten data
    X_train = stack_s(X_train)
    X_val = stack_s(X_val)
    X_test = stack_s(X_test)
    
    #$ Transpose dims so sample is axis 0
    y_train = y_train.transpose('sample', 'lat', 'lon')
    y_val = y_val.transpose('sample', 'lat', 'lon')
    y_test = y_test.transpose('sample', 'lat', 'lon')
    
    #$ Chunking
    print(f"{datetime.now() - START_TIME}\t Chunking data")
    chunks = {'sample': 3650, 's': 10_000}
    X_train = X_train.chunk(chunks)
    X_val = X_val.chunk(chunks)
    X_test = X_test.chunk(chunks)
    chunks = {'sample': 3650}
    y_train = y_train.chunk(chunks)
    y_val = y_val.chunk(chunks)
    y_test = y_test.chunk(chunks)
    
    #$ Paths for saving
    print(f"{datetime.now() - START_TIME}\t Saving to {DATA_PATH}")
    os.makedirs(DATA_PATH, exist_ok=True)
    
    #$ Save data
    print(f"{datetime.now() - START_TIME}\t Save data")
    #$$ Standardizers
    utils.processing.save_standardizer(
        X_standardizer, DATA_PATH, name='X_standardizer')
    utils.processing.save_standardizer(
        y_standardizer, DATA_PATH, name='y_standardizer')
    
    #$$ Features
    print(f"{datetime.now() - START_TIME}\t Save features")
    with ProgressBar():
        utils.data.save_flattened_features(X_train, 'train', DATA_PATH)
        utils.data.save_flattened_features(X_val, 'val', DATA_PATH)
        utils.data.save_flattened_features(X_test, 'test', DATA_PATH)
    
    #$$ Targets
    print(f"{datetime.now() - START_TIME}\t Save targets")
    with ProgressBar():
        utils.data.save_flattened_targets(y_train, 'train', DATA_PATH)
        utils.data.save_flattened_targets(y_val, 'val', DATA_PATH)
        utils.data.save_flattened_targets(y_test, 'test', DATA_PATH)
        
    #$ Save pytorch tensors
    print(f"{datetime.now() - START_TIME}\t Save tensors as single-precision float")
    # Load dataarrays
    X_train = utils.data.load_flattened_features(DATA_PATH, 'train')
    y_train = utils.data.load_flattened_targets(DATA_PATH, 'train')
    X_val = utils.data.load_flattened_features(DATA_PATH, 'val')
    y_val = utils.data.load_flattened_targets(DATA_PATH, 'val')
    X_test = utils.data.load_flattened_features(DATA_PATH, 'test')
    y_test = utils.data.load_flattened_targets(DATA_PATH, 'test')
    
    X_train = torch.from_numpy(X_train.astype(np.float32).values)
    y_train = torch.from_numpy(y_train.astype(np.float32).values)
    X_val = torch.from_numpy(X_val.astype(np.float32).values)
    y_val = torch.from_numpy(y_val.astype(np.float32).values)
    X_test = torch.from_numpy(X_test.astype(np.float32).values)
    y_test = torch.from_numpy(y_test.astype(np.float32).values)
    
    torch.save(X_train, os.path.join(DATA_PATH, 'X_train_tensor.pt'))
    torch.save(y_train, os.path.join(DATA_PATH, 'y_train_tensor.pt'))
    torch.save(X_val, os.path.join(DATA_PATH, 'X_val_tensor.pt'))
    torch.save(y_val, os.path.join(DATA_PATH, 'y_val_tensor.pt'))
    torch.save(X_test, os.path.join(DATA_PATH, 'X_test_tensor.pt'))
    torch.save(y_test, os.path.join(DATA_PATH, 'y_test_tensor.pt'))
    
    print("PROCESS COMPLETED")
    print(datetime.now() - START_TIME)
    return 0

if __name__ == "__main__":
    main()
