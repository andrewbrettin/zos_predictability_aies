__all__ = [
    'get_lat_lon_from_icoord',
    'get_preds',
]

import os
import json
import numpy as np
import xarray as xr

import utils

#$ Global variables
with open("~/s2s/paths.json") as paths_json:
    PATHS = json.load(paths_json)
with open("~/s2s/globals.json") as globals_json:
    GLOBALS = json.load(globals_json)

def get_lat_lon_from_icoord(icoord):
    # Necessary data
    icoords_all = np.load(os.path.join(PATHS['s2s_predictions'], 'icoords.npy'))
    prediction_path = os.path.join(PATHS['s2s_predictions'], 'tau_20')
    target_template = xr.open_dataarray(
        os.path.join(prediction_path, 'target.nc')).isel(time=0)

    ilat, ilon = icoords_all[icoord]
    lat = target_template.isel(lat=ilat//2)['lat'].item()
    lon = target_template.isel(lon=ilon//2)['lon'].item()

    return lat, lon

def get_icoord_from_lat_lon(lat, lon):
    """
    Gets icoord from approximate latitude and longitude.
    """
    # Load necessary datasets
    icoords_all = np.load(os.path.join(PATHS['s2s_predictions'], 'icoords.npy'))
    prediction_path = os.path.join(PATHS['full_globe'], 'predictions', 'tau_20')
    target_template = xr.open_dataarray(
        os.path.join(prediction_path, 'target.nc')).isel(time=0)

    # Get adjusted latitudes and longitudes
    lats = target_template['lat']
    lons = target_template['lon']
    adjusted_lat = lats.sel(lat=lat, method='nearest').item()
    adjusted_lon = lons.sel(lon=lon, method='nearest').item()

    # Compute integer latitude and longitude indices
    ilat = np.where(lats == adjusted_lat)[0][0]
    ilon = np.where(lons == adjusted_lon)[0][0]

    icoord = np.where((icoords_all[:,0] == 2*ilat) * (icoords_all[:,1] == 2*ilon))[0][0]

    return icoord

def get_preds(tau, point=None):
    prediction_path = os.path.join(PATHS['full_globe'], 'predictions', f'tau_{tau}')

    # Get predictions
    pred_mean = xr.open_dataarray(os.path.join(prediction_path, 'pred_mean.nc'))
    pred_logvar = xr.open_dataarray(os.path.join(prediction_path, 'pred_logvar.nc'))

    # Select point
    if point is not None:
        pred_mean = pred_mean.sel(point)
        pred_logvar = pred_logvar.sel(point)

    return pred_mean, pred_logvar