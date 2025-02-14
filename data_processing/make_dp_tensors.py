r"""
make_dp_tensors.py

Gets damped persistence features. See also: s2s/archive/tensor_configs/make_data_0.py

```
grep -rin --include=\*.py 'utils.data.save_flattened_dp_features' ./
```

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
T = 5    # Predict 5-day averages
TAU = int(sys.argv[1])

#$ Paths
DATA_PATH = os.path.join(PATHS['full_globe'], 'data', f'tau_{TAU}')

#$ Functions
def create_dp_datasets(tau, T):
    X_dp_data = []
    y_data = []
    all_members = list(product(GLOBALS['init_years'], GLOBALS['members']))
    
    for i, (init_year, member) in enumerate(all_members):
        #$ Make targets
        #$$ Uncoarsened dynamic sea level anomalies
        zos = utils.data.load_anomalies('zos', init_year, member, chunkedby='space')
        
        # Select points between (-60, 60) only
        zos = zos.sel(lat=slice(-60,60))

        times = (
            zos['time']
            .isel(time=slice(T-1, None, T))
            .shift(time=-tau//T)
            .isel(time=slice(0, -tau//T))
            .time
            .values
        )
        
        X = (
            zos.rolling(time=T, center=False).mean()
            .sel(time=times)
        )

        #$ Rename
        X.name = f"{X.name}_{i}"

        X.attrs['init_year'] = init_year
        X.attrs['member'] = member

        #$ Append variables
        X_dp_data.append(X)
        
    #$ Merge variables
    X_dp_all = xr.merge(X_dp_data)
    X_dp_all = X_dp_all.to_array(dim='member_id')
    X_dp_all['member_id'] = np.arange(len(X_dp_data))

    #$ Create sample dimension to reduce dimensionality
    X_dp_all = X_dp_all.stack(sample=('member_id', 'time'))
    
    return X_dp_all

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
        X_dp_all = create_dp_datasets(TAU, T)

    print(f"{datetime.now() - START_TIME}\t Splitting into train/val/test")
    with ProgressBar():
        X_dp_train, X_dp_val, X_dp_test = get_train_test_val(X_dp_all)

    
    # Load standardizer for sea level data
    X_dp_standardizer = utils.processing.load_standardizer(DATA_PATH, datatype='y')

    # Standardizing
    print(f"{datetime.now() - START_TIME}\t Standardizing")
    with ProgressBar():
        X_dp_train = utils.processing.standardize(X_dp_train, X_dp_standardizer)
        X_dp_val = utils.processing.standardize(X_dp_val, X_dp_standardizer)
        X_dp_test = utils.processing.standardize(X_dp_test, X_dp_standardizer)
    
    #$ Transpose dims so sample is axis 0
    X_dp_train = X_dp_train.transpose('sample', 'lat', 'lon')
    X_dp_val = X_dp_val.transpose('sample', 'lat', 'lon')
    X_dp_test = X_dp_test.transpose('sample', 'lat', 'lon')
    
    #$ Chunking
    print(f"{datetime.now() - START_TIME}\t Chunking data")
    chunks = {'sample': 3650}
    X_dp_train = X_dp_train.chunk(chunks)
    X_dp_val = X_dp_val.chunk(chunks)
    X_dp_test = X_dp_test.chunk(chunks)

    #$ Paths for saving
    print(f"{datetime.now() - START_TIME}\t Saving to {DATA_PATH}")
    os.makedirs(DATA_PATH, exist_ok=True)

    #$$ Features
    print(f"{datetime.now() - START_TIME}\t Save damped persistence features")
    with ProgressBar():
        utils.data.save_flattened_dp_features(X_dp_train, 'train', DATA_PATH)
        utils.data.save_flattened_dp_features(X_dp_val, 'val', DATA_PATH)
        utils.data.save_flattened_dp_features(X_dp_test, 'test', DATA_PATH)

    print("PROCESS COMPLETED")
    print(datetime.now() - START_TIME)
    return 0

if __name__ == "__main__":
    main()




