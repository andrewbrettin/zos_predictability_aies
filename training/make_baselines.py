#$ Imports
import os
import shutil
import sys
import json
from datetime import datetime
import warnings

import numpy as np
from scipy import stats, linalg
import xarray as xr
import cftime
from tqdm import tqdm
import properscoring as ps
from sklearn.linear_model import LogisticRegression

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cmocean as cmo

import torch
from utils import helpers
import utils

warnings.filterwarnings('ignore')

#$ Global variables
with open("~/s2s/paths.json") as paths_json:
    PATHS = json.load(paths_json)
with open("~/s2s/globals.json") as globals_json:
    GLOBALS = json.load(globals_json)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(f"Using cuda device {torch.cuda.get_device_name(0)}")

TAU = int(sys.argv[1])
START_TIME = datetime.now()

def compute_dp_coef(X, y):
    if np.any(np.isnan(X)):
        return np.nan
    return np.dot(X, y) / linalg.norm(X)**2
    
def logistic_regression(X, y):
    if np.any(np.isnan(X)):
        return np.nan, np.nan
    model = LogisticRegression()
    model.fit(X[:, np.newaxis], y[:, np.newaxis])
    
    return model.intercept_.item(), model.coef_.item()

def main():
    prediction_path = os.path.join(PATHS['full_globe'], 'predictions', f'tau_{TAU}')
    os.makedirs(prediction_path, exist_ok=True)
    
    print(f'tau = {TAU}', datetime.now() - START_TIME)
    # Load data and subsample lat/lon x2
    X_train = xr.open_dataarray(os.path.join(PATHS['full_globe'], 'data', f'tau_{TAU}', 'X_dp_train.nc'))
    X_train = X_train[:, ::2, ::2]
    y_train = xr.open_dataarray(os.path.join(PATHS['full_globe'], 'data', f'tau_{TAU}', 'y_train.nc'))
    y_train = y_train[:, ::2, ::2]

    # Load test data, resample, and make dims ('time', 'lat', 'lon') instead of ('sample', 'lat', 'lon')
    X_test = xr.open_dataarray(os.path.join(PATHS['full_globe'], 'data', f'tau_{TAU}', 'X_dp_test.nc'))
    X_test = X_test[:, ::2, ::2]
    times = xr.CFTimeIndex(X_test['time'].values)
    lat = X_test['lat']
    lon = X_test['lon']
    X_test = xr.DataArray(
        X_test.values, dims=('time', 'lat', 'lon'),
        coords={'time': times, 'lat': lat, 'lon': lon}
    )

    # Compute Damped Persistence coefficients
    beta = xr.apply_ufunc(
        compute_dp_coef,
        X_train,
        y_train,
        input_core_dims=[['sample'], ['sample']],
        output_core_dims=[[]],
        vectorize=True
    )
    # Save damped persistence predictions
    dp_pred = X_test * beta
    beta.to_netcdf(os.path.join(prediction_path, 'dp_coef.nc'))
    dp_pred.to_netcdf(os.path.join(prediction_path, 'dp_pred.nc'))

    # Logistic regression baseline targets
    is_positive = (y_train > 0).astype(float)
    # Compute logistic regression coefficients
    beta_0, beta_1 = xr.apply_ufunc(
        logistic_regression,
        X_train,
        is_positive,
        input_core_dims=[['sample'], ['sample']],
        output_core_dims=[[], []],
        # exclude_dims={'lat', 'lon'},
        vectorize=True
    )
    coef_ds = xr.Dataset({'beta_0': beta_0, 'beta_1': beta_1})
    # Save logistic_regression predictions
    coef_ds.to_netcdf(os.path.join(prediction_path, 'logreg_pos_coefs.nc'))
    prob_pos = 1/(1 + np.exp(-(beta_0 + beta_1*X_test)))
    prob_pos.to_netcdf(os.path.join(prediction_path, 'logreg_prob_pos.nc'))

    return 0

if __name__ == "__main__":
    main()

