"""
coarsen.py

This script coarsens a variable.

Right now this script is configured to load SST anomalies, but we may 
consider modifying the script in the future to load other variables or
the raw (non-deseasonalized) data.

Some quick checks have shown that there is little difference between
deseasonalizing and then coarsening or coarsening and then
deseasonalizing.
"""

__author__ = "@andrewbrettin"

import os
import sys
import json
from datetime import datetime
from itertools import product
import numpy as np
import xarray as xr
import xesmf as xe
import dask
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
sys.path.append('..')
import utils

#$ Globals
with open('~/s2s/paths.json') as paths_json:
    PATHS = json.load(paths_json)
with open('~/s2s/globals.json') as globals_json:
    GLOBALS = json.load(globals_json)
    
VARIABLES = ['SST', 'zos', 'UBOT', 'VBOT']
RESOLUTION = 5

def save_name(init_year, member, var):
    return f'LE2-{init_year}.{member}.{var}_anom.zarr'

def main():
    START_TIME = datetime.now()
    print(f"Beginning {__file__}")
    print(START_TIME)
    
    #$ Make paths
    path = os.path.join(PATHS['coarsened'], str(RESOLUTION))
    regridder_path = os.path.join(path, 'grid')
    coarsened_path = os.path.join(path, 'anomalies')
    os.makedirs(path, exist_ok=True)
    os.makedirs(regridder_path, exist_ok=True)
    os.makedirs(coarsened_path, exist_ok=True)
    
    #$ Loop
    for init_year, member, var in product(
            GLOBALS['init_years'], GLOBALS['members'], VARIABLES):
        
        ds_name = save_name(init_year, member, var)
        print(f"{datetime.now() -  START_TIME}\t {ds_name}")
        
        arr = utils.data.load_anomalies(var, init_year, member)
        ds_in = xr.Dataset({var: arr})
        
        coarsened_ds = utils.processing.coarsen(
            ds_in, res=RESOLUTION, regridder_path=regridder_path)
        
        #$ Fix metadata
        coarsened_ds.attrs['history'] = (
            f"Created on {datetime.now()} using {__file__}"
        )
        
        #$ Save
        print(f"{datetime.now() -  START_TIME}\t Saving to {coarsened_path}")
        coarsened_ds.to_zarr(os.path.join(coarsened_path, ds_name))
    
    return 0

def test():
    init_year = 1301
    member = '013'
    var = 'zos'
    
    START_TIME = datetime.now()
    print(f"Beginning {__file__}")
    print(START_TIME)
    
    #$ Make paths
    path = os.path.join(PATHS['coarsened'], str(RESOLUTION))
    regridder_path = os.path.join(path, 'grid')
    coarsened_path = os.path.join(path, 'anomalies')
    os.makedirs(path, exist_ok=True)
    os.makedirs(regridder_path, exist_ok=True)
    os.makedirs(coarsened_path, exist_ok=True)
    
        
    ds_name = save_name(init_year, member, var)
    print(f"{datetime.now() -  START_TIME}\t {ds_name}")
    
    arr = utils.data.load_anomalies(var, init_year, member)
    ds_in = xr.Dataset({var: arr})
    
    coarsened_ds = utils.processing.coarsen(
        ds_in, res=RESOLUTION, regridder_path=regridder_path)
    
    #$ Fix metadata
    coarsened_ds.attrs['history'] = (
        f"Created on {datetime.now()} using {__file__}"
    )
    
    #$ Save
    print(f"{datetime.now() -  START_TIME}\t Saving to {coarsened_path}")
    with ProgressBar():
        coarsened_ds.to_zarr(os.path.join(coarsened_path, ds_name))
    
    return 0

if __name__ == "__main__":
    test()
    # main()
