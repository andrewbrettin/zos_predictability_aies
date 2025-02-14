"""
processing.py

Various functions for processing data: Regridding, coarsening, standardizing,
deseasonalizing, etc.
"""
__author__ = "@andrewbrettin"
__all__ = [
    "create_lat_lon_from_resolution",
    "create_areas",
    "get_regridder_filename",
    "create_regridder",
    "regrid",
    "detrend",
    "deseasonalize",
    "rm_stores",
    "execute_rechunk",
    "get_standardizer",
    "standardize",
    "unstandardize",
    "save_standardizer",
    "load_standardizer",
    "correct_cftimes",
]
import os
import json
import sys
import shutil
from datetime import datetime
from time import strptime
import numpy as np
import scipy as sc
import pandas as pd
import xarray as xr
import xesmf as xe
import cftime
import dask
from rechunker import rechunk
import matplotlib.pyplot as plt

#$ Globals
with open('~/s2s/paths.json') as paths_json:
    PATHS = json.load(paths_json)    
with open('~/s2s/globals.json') as globals_json:
    GLOBALS = json.load(globals_json)


#$ Functions
#$ Regridding
def create_lat_lon_from_resolution(res: float) -> (xr.DataArray, xr.DataArray):
    """
    Given a resolution, returns lats and lons dataarrays 
    corresponding to that resolution.
    
    Parameters:
        res: float
            Target resolution, in degrees.
    Returns:
        lats: xr.DataArray
            Target latitudes
        lons: xr.DataArray
            Target longitudes
    """
    lats_vals = np.arange(-90, 90, res)
    lats = xr.DataArray(lats_vals, dims=('lat',), coords={'lat': lats_vals})
    
    lons_vals = np.arange(0, 360, res)
    lons = xr.DataArray(lons_vals, dims='lon', coords={'lon': lons_vals})
    
    return lats, lons

def create_areas(lats, lons):
    """
    Creates area files based on given latitudes and longitudes.
    
    Parameters:
        lats: xr.DataArray
            Target latitudes
        lons: xr.DataArray
            Target longitudes
            
    Returns:
        areas: xr.DataArray
            Grid cell areas.
    """
    #$ Latitude and longitude spacing, in degrees
    dlats = lats[1:].values - lats[:-1].values
    dlons = lons[1:].values - lons[:-1].values
    dlat = dlats.mean()
    dlons = dlons.mean()
    
    R_e = 6378.1
    dx = np.pi * R_e * np.cos(lats * np.pi/180) * dlons / 360
    dy = 2 * np.pi * R_e * dlat / 360 * xr.ones_like(lons)
    areas = dx * dy
    
    #$ Metadata
    areas.name = 'area'
    areas.attrs['units'] = 'km^2'
    areas.attrs['long_name'] = 'grid_cell_areas'
    areas.attrs['history'] = (
        f"Created on {datetime.now()}"
    )
    return areas
    
def get_regridder_filename(ds_in, ds_out, method='bilinear', periodic=True):
    """
    Based on xesmf.Regridder._get_default_filename()
    Copied so that we can specify output path.
    """
    filename = "{}_{}x{}_{}x{}".format(
        method,
        len(ds_in['lat']),
        len(ds_in['lon']),
        len(ds_out['lat']),
        len(ds_out['lon'])
    )
    if periodic:
        filename += '_peri.nc'
    else:
        filename += '.nc'
    return filename
        
    
def create_regridder(ds_in, ds_out, method='bilinear', periodic=True, 
                     save_path=None):
    """
    Creates regridder weights.
    """
    #$ Create filename
    filename = get_regridder_filename(
        ds_in, 
        ds_out,
        method=method,
        periodic=periodic
    )
    
    if save_path is None:
        save_path = ''
    
    filename_full = os.path.join(
        save_path,
        filename
    )
        
    #$ Input grid
    regridder = xe.Regridder(
        ds_in, ds_out, method, periodic=periodic, filename=filename_full
    )
    
    return regridder

def coarsen(ds_in, res, regridder_path=None):
    """
    Coarsens the dataset to a specified resolution.
    
    Parameters:
        ds_in: xr.Dataset
            Input dataset
        res: float
            Output resolution
        regridder_path: str
            Output path to save regridder file.
    """
    assert isinstance(ds_in, xr.Dataset)
    
    out_lat, out_lon = create_lat_lon_from_resolution(res)
    ds_out = xr.Dataset({'lat': out_lat, 'lon': out_lon})
    
    if regridder_path is None:
        regridder_path = ''
    
    regridder = create_regridder(
        ds_in, 
        ds_out, 
        method='bilinear',
        periodic=True,
        save_path=regridder_path
    )
    
    regridded_ds = regridder(ds_in, keep_attrs=True)
    return regridded_ds

def detrend(arr, dim='time', deg=5):
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        p = arr.polyfit(dim=dim, deg=deg)
    p['polyfit_coefficients'] = p['polyfit_coefficients'].chunk(
        {'lat': 48, 'lon': 48}
    )
    trend = xr.polyval(arr[dim], p['polyfit_coefficients'])
    detrended = arr - trend
    
    detrended.name = arr.name
    detrended.attrs['units'] = arr.attrs['units']
    detrended.attrs['description'] = (
        f"Detrended {arr.name} using a degree {deg} polynomial."
    )
    detrended.attrs['history'] = (
        f"Created on {datetime.now()} using {FILE}"
    )
    return detrended

def deseasonalize(arr):
    if len(arr['time']) == 0:
        return arr
    gb = arr.groupby('time.dayofyear')
    clim = gb.mean(dim='time')
    anom = (gb - clim).drop_vars('dayofyear')
    return anom

#$ File management
def rm_stores(*stores):
    for store in stores:
        if os.path.exists(store):
            shutil.rmtree(store)

#$ Rechunking
def execute_rechunk(ds, target_store, temp_store, chunkedby='time'):
    if chunkedby == 'time':
        chunks_dict = {'time': -1, 'lat': 48, 'lon': 48}
    if chunkedby == 'space':
        chunks_dict = {'time': 3650, 'lat': -1, 'lon': -1}
    
    max_mem='8GB'
    
    array_plan = rechunk(
        ds, chunks_dict, max_mem, target_store, temp_store=temp_store
    )
    
    array_plan.execute()
    
#$ Standardization
def get_standardizer(da, dim=None):
    """
    Returns a dict of the mean and standard dev of ds.
    
    If there are multiple variables, return a xarray.DataArray. Otherwise, 
    returns a dictionary.
    """
    if dim is not None:
        mean = da.mean(dim=dim)
        std = da.std(dim=dim)
        standardizer = xr.Dataset({'mean': mean, 'std': std})
    elif 'var' in da.dims:
        mean = da.mean(dim=('lat', 'lon', 'sample'))
        std = da.std(dim=('lat', 'lon', 'sample'))
        standardizer = xr.Dataset({'mean': mean, 'std': std})
    else:
        mean = da.mean().compute().item()
        std = da.std().compute().item()
        standardizer = dict(mean=mean, std=std)
    return standardizer

def standardize(da, standardizer=None):
    if standardizer is None:
        standardizer = get_standardizer(da)
        
    da = (da - standardizer['mean']) / standardizer['std']
    return da

def unstandardize(da, standardizer):
    return standardizer['mean'] + standardizer['std'] * da

def save_standardizer(standardizer, data_path=None, name=None):
    """
    Saves standardizer.
    
    Parameters:
        standardizer: dict
            Standardizer dictionary to pickle.
        data_path: str
            Path to save the data.
        name: str
            Name of standardizer file. Extension `.npy` or `.nc` is
            added automatically.
    """
    if data_path is None:
        data_path = ''
    if name is None:
        name = 'standardizer'
    if isinstance(standardizer, dict):
        ext = '.npy'
        name = name + ext
        np.save(os.path.join(data_path, name), standardizer)
    elif isinstance(standardizer, xr.Dataset):
        standardizer.load()
        ext = '.nc'
        name = name + ext
        standardizer.to_netcdf(os.path.join(data_path, name), compute=True)
    
def load_standardizer(data_path, datatype='X'):
    """
    Loads standardizer
    
    Parameters:
        data_path: str
            Path to save the data.
        datatype: str
            X for feature standardizer,
            y for target standardizer.
    Returns:
        standardizer: dict
            Standardizer dictionary.
    """
    assert datatype in ['X', 'y', 'residual']
    np_filename = os.path.join(data_path, f'{datatype}_standardizer.npy')
    xr_filename = os.path.join(data_path, f'{datatype}_standardizer.nc')
    if os.path.exists(np_filename):
        standardizer = np.load(np_filename, allow_pickle=True).item()
    elif os.path.exists(xr_filename):
        standardizer = xr.open_dataset(xr_filename)
    else:
        raise OSError(f'Standardizer not found in {data_path}')
    return standardizer

def to_cftime(stringtimes, format="%Y-%m-%d %H:%M:%S", 
                          calendar='standard'):
    """
    Given a xr.DataArray da which has times given as strings, corrects
    the time dimension so that the times have datatype cftime.
    
    Note: Should be deprecated in favor of cftime_strptime and 
    datestrings_to_cftimeindex
    """
    if isinstance(stringtimes, xr.DataArray):
        stringtimes = stringtimes.values
    cftime_data = [
        cftime.datetime.strptime(s, format=format, calendar=calendar)
        for s in stringtimes
    ]
    cftimes = xr.DataArray(
        cftime_data,
        dims=('time',),
        coords={'time': cftime_data},
        name='time'
    )
    return cftimes

def correct_cftimes(da, format="%Y-%m-%d %H:%M:%S", calendar='standard'):
    """
    Given a xr.DataArray da which has times given as strings, corrects
    the time dimension so that the times have datatype cftime.
    
    Note: Should be deprecated in favor of cftime_strptime and 
    datestrings_to_cftimeindex
    """
    cftimes = to_cftime(da['time'])
    da['time'] = cftimes
    return da


def cftime_strptime(date_string, format='%Y-%m-%d %H:%M:%S', calendar='standard'):
    args = strptime(date_string, format)[0:6]
    if calendar=='standard' or calendar=='gregorian':
        datetime = cftime.DatetimeGregorian(*args)
    elif calendar=='proleptic_gregorian':
        datetime = cftime.DatetimeProlepticGregorian(*args)
    elif calendar=='noleap' or calendar=='365_day':
        datetime = cftime.DatetimeNoLeap(*args)
    elif calendar=='360_day':
        datetime = cftime.Datetime360Day(*args)
    elif calendar=='julian':
        datetime = cftime.DatetimeJulian(*args)
    elif calendar=='allleap' or calendar=='366_day':
        datetime = cftime.DatetimeAllLeap(*args)
    else:
        raise ValueError(
            f"Invalid calendar '{calendar}'. Valid calendars are: "
            "'standard', 'gregorian', 'proleptic_gregorian', 'noleap', "
            "'365_day', '360_day', 'julian', 'all_leap', '366_day'."
        )
    return datetime

def datestrings_to_cftimeindex(date_string, format='%Y-%m-%d %H:%M:%S', calendar='standard'):
    f = np.vectorize(cftime_strptime)
    dates_array = f(date_string, format=format, calendar=calendar)
    return xr.CFTimeIndex(dates_array)