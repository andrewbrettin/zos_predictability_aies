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
TAU = int(sys.argv[1])

DATA_PATH = os.path.join(PATHS['full_globe'], 'data', f'tau_{TAU}')

def get_ocean_indices(y_test):
    isnan = torch.any(torch.isnan(y_test), axis=0)
    nlats, nlons = isnan.shape
    icoords = np.array(list(product(np.arange(nlats), np.arange(nlons))))
    ocean_points = [bool(~isnan[ilat, ilon]) for ilat, ilon in icoords]
    icoords = icoords[ocean_points]
    return icoords

def main():
    print(datetime.now())
    print(__file__)
    print(DATA_PATH)
    
    y_test_tensor = torch.load(os.path.join(DATA_PATH, 'y_test_tensor.pt'))
    
    # Get ocean indices
    icoords = get_ocean_indices(y_test_tensor)
    
    # Subselect rows where both ilat and ilon are even
    even_icoords = [pair for pair in icoords if ((pair[0] % 2 == 0) and (pair[1] % 2 == 0))]
    even_icoords = np.array(even_icoords)
    
    
    # Print stuff
    print("ilats, ilons [0:5]:", even_icoords[0:10, :])
    print("icoords shape:", even_icoords.shape)
    
    np.save(os.path.join(DATA_PATH, 'icoords.npy'), even_icoords)
    
    return 0

if __name__ == "__main__":
    main()