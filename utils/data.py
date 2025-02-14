"""
data.py

Various utility functions for loading data.
"""

__author__ = "@andrewbrettin"
__all__ = [
    "load_areas",
    "load_mask",
    "load_dataset",
    "load_anomalies",
    "select_region",
    "save_flattened_features",
    "save_flattened_targets",
    "load_s_coord",
    "load_flattened_features",
    "load_flattened_targets",
    "load_coarsened_anomalies"
]

import os
import sys
import json
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client

with open("~/s2s/paths.json") as paths_json:
    PATHS = json.load(paths_json)
with open("~/s2s/globals.json") as globals_json:
    GLOBALS = json.load(globals_json)

def load_areas():    
    areas = xr.open_dataset(os.path.join(PATHS['grid'], 'areas.nc'))['area']
    return areas

def load_mask():
    mask = xr.open_dataset(os.path.join(PATHS['grid'], 'mask.nc'))['land_mask']
    return mask

def load_dataset(var, init_year, member, full_ds=False, chunkedby='space'):
    """
    Loads a given dataarray on the lat-lon grid.
    For ocean variables, the land points are masked by np.nan.
    For atmospheric variables, there for some reason is a one-day
    overlap on 01-01-2015 where fields for both the BHIST and SSP370
    forcing scenarios occur. Therefore, we need to properly select
    dates 01-01-1850 through 12-31-2014 for the historical emissions
    and dates 01-01-2015 through 12-31-2100 for SSP370.
    
    Parameters:
        var: str
            Variable to load. Should be one of the following: 'SSH', 'zos', 
            'SSH_2', 'SST', 'PSL', 'UBOT', 'VBOT', 'SHFLX'.
        member: str
            Ensemble member.
        init_year: int or str
            Initialization year.
        full_ds: bool
            If full_ds is True, then returns an xr.Dataset object.
            Otherwise, it returns the xr.DataArray object.
        chunkedby: str, 'space' or 'time'
            Indicates whether to load datasets chunked by space or by time.
    Returns
        arr: xr.Dataset or xr.DataArray
            DataArray object dataset on the lat-lon grid. If var is
            an ocean variable, land points are masked with np.nan.
    """

    if chunkedby == 'time':
        path = os.path.join(
            PATHS['rechunked'], 
            f'LE2-{init_year}.{member}.{var}_rechunked.zarr'
        )
        ds = xr.open_zarr(path, consolidated=False)
        arr = ds[var]
    elif chunkedby == 'space':
        if var in ['PSL', 'UBOT', 'VBOT', 'SHFLX']:
            path = os.path.join(
                PATHS['atm_daily'], var,
                f'b.e21*.f09_g17.LE2-{init_year}.{member}.cam.h1.{var}.*.nc'
            )
            ds = xr.open_mfdataset(path, combine='by_coords')
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = ds.drop_duplicates(dim='time', keep='last')
            ds = ds.chunk({'time': 3650})
            arr = ds[var]

        elif var in ['SSH_2', 'SST', 'SHF', 'MLD']:
            path = os.path.join(
                PATHS['regridded'], f'LE2-{init_year}.{member}.{var}.nc'
            )
            ds = xr.open_dataset(path, chunks={'time':3650})
            mask = load_mask()
            arr = xr.where(mask, ds[var], np.nan)
            arr.attrs = ds[var].attrs
            ds[var] = arr

        elif var in ['zos', 'SSH']:
            path = os.path.join(
                PATHS['sea_level'], f'LE2-{init_year}.{member}.{var}.zarr'
            )
            ds = xr.open_zarr(path)
            mask = load_mask()
            arr = xr.where(mask, ds[var], np.nan)
            arr.attrs = ds[var].attrs
            ds[var] = arr

        else:
            raise ValueError(
                f"Incorrect variable {var}. Should be one of: "
                "['SSH', 'zos', 'SSH_2', 'SST', 'PSL', 'UBOT', 'VBOT',"
                "'SHFLX', 'QHF_2']"
            )
    else:
        raise ValueError(
            "Parameter 'chunkedby' must take values 'space' or 'time'"
        )
        
    if full_ds:
        return ds
    else:
        return arr
    
def load_anomalies(var, init_year, member, full_ds=False, chunkedby='space'):
    if chunkedby == 'time':
        path = os.path.join(
            PATHS['detrended_deseasonalized'], 
            f'LE2-{init_year}.{member}.{var}_anom.zarr'
        )
        ds = xr.open_zarr(path, consolidated=False)
    elif chunkedby == 'space':
        path = os.path.join(
            PATHS['anom_spatial'], 
            f'LE2-{init_year}.{member}.{var}_anom.zarr'
        )
        ds = xr.open_zarr(path, consolidated=False)

    if full_ds:
        return ds
    else:
        return ds[var]
    
def load_coarsened_anomalies(var, res, init_year, member, full_ds=False, 
                             chunkedby='space'):
    """
    Loads coarsened anomalies chunked in space.
    
    Parameters:
        var: str
            Variable to load
        res: float
            resolution to load
        init_year: int or str
            Initialization year.
        member: str
            Ensemble member.
        full_ds: bool
            If full_ds is True, then returns an xr.Dataset object.
            Otherwise, it returns the xr.DataArray object.
        chunkedby: 'space' or 'time'
            Whether to load data chunked by space or time.
            Only is used if res == 1.
            
    """
    # Special case if resolution == 1
    if res == 1:
        anom = load_anomalies(
            var, init_year, member, full_ds=full_ds, chunkedby=chunkedby)
        return anom
    else:
        filename = f'LE2-{init_year}.{member}.{var}_anom.zarr'
        path = os.path.join(
            PATHS['coarsened'],
            str(res),
            'anomalies'
        )
        file = os.path.join(path, filename) 
        ds = xr.open_zarr(file, consolidated=False)
        if full_ds:
            return ds
        else:
            return ds[var]
    
def select_region(darray, region):
    """
    Selects coordinates from a given dataarray for a few 
    prespecified regions.
    
    Params:
        darray: xr.DataArray or xr.Dataset
            DataArray or Dataset to subset.
        region: str
            Region to subset. See globals.json for valid
            regions.
    Returns:
        subset: xr.DataArray or xr.Dataset
            Subsetted dataarray/dataset.
    """
    coords_dict = GLOBALS['regions'][region]
    subset = darray.sel(
        lat=slice(*coords_dict['lat']),
        lon=slice(*coords_dict['lon'])
    )
    return subset

def save_flattened_features(X_ds, datatype, path):
    assert datatype in ['train', 'test', 'val']
        
    if 'sample' in X_ds.indexes:
        #$ Save sample dim coordinates as csv
        sample_df = X_ds.indexes['sample'].to_frame().reset_index(drop=True)
        sample_df.to_csv(
            os.path.join(path, f'X_{datatype}_sample_coord.csv'),
            index=False
        )
        X_ds = X_ds.reset_index('sample')
    
    if 's' in X_ds.indexes:
        #$ Save s coordinates as csv
        s_df = X_ds.indexes['s'].to_frame().reset_index(drop=True)
        s_df.to_csv(
            os.path.join(path, f'X_{datatype}_s_coord.csv'),
            index=False
        )
        X_ds = X_ds.reset_index('s')
    
    #$ Convert to xr.Dataset if needed
    if isinstance(X_ds, xr.DataArray):
        X_ds = xr.Dataset({'X': X_ds})
        
    X_ds.to_netcdf(os.path.join(path, f'X_{datatype}.nc'), mode='w')

def save_flattened_targets(y_ds, datatype, path):
    assert datatype in ['train', 'test', 'val']
    
    if 'sample' in y_ds.indexes:
        sample_df = y_ds.indexes['sample'].to_frame().reset_index(drop=True)
        sample_df.to_csv(
            os.path.join(path, f'y_{datatype}_sample_coord.csv'),
            index=False
        )
        y_ds = y_ds.reset_index('sample')
    
    if isinstance(y_ds, xr.DataArray):
        y_ds = xr.Dataset({'y': y_ds})
    y_ds.to_netcdf(os.path.join(path, f'y_{datatype}.nc'), mode='w')
    
def save_flattened_dp_features(X_dp_ds, datatype, path):
    assert datatype in ['train', 'test', 'val']
    
    if 'sample' in X_dp_ds.indexes:
        # Should be unnecessary to save coordinate info
        # <...>
        # Reset index
        X_dp_ds = X_dp_ds.reset_index('sample')
    
    if isinstance(X_dp_ds, xr.DataArray):
        X_dp_ds = xr.Dataset({'X_dp': X_dp_ds})
    X_dp_ds.to_netcdf(os.path.join(path, f'X_dp_{datatype}.nc'), mode='w')

def load_s_coord(path, datatype):
    """Loads spatial coordinate s for a given dataset."""
    s_filename = os.path.join(path, f'X_{datatype}_s_coord.csv')
    if os.path.exists(s_filename):
        s_df = pd.read_csv(s_filename)
        s_multiindex = pd.MultiIndex.from_frame(s_df)
        s_multiindex.name = 's'
        s = xr.DataArray(s_multiindex)
        return s
    else:
        raise FileNotFoundError(f"No such file {path}")
    
def load_flattened_features(path, datatype):
    """
    Loads flattened features. Features should contain two multiindexes:
    'sample', which is composed of times and ensemble member numbers;
    and 's', which contains the spatial variables lat and lon.
    
    Parameters:
        path: str
            Path to features.
        datatype: str
            dataname of the form 'train', 'val', or 'test'
    Returns:
        X: xr.DataArray
            The flattened features with correct indexes.
    """
    assert datatype in ['train', 'val', 'test']
    X = xr.open_dataset(os.path.join(path, f'X_{datatype}.nc'))
    
    sample_filename = os.path.join(path, f'X_{datatype}_sample_coord.csv')
    if os.path.exists(sample_filename):
        sample_df = pd.read_csv(sample_filename)
        sample_multiindex = pd.MultiIndex.from_frame(sample_df)
        sample_multiindex.name = 'sample'
        sample = xr.DataArray(sample_multiindex)
        X['sample'] = sample
    
    s_filename = os.path.join(path, f'X_{datatype}_s_coord.csv')
    if os.path.exists(s_filename):
        s_df = pd.read_csv(s_filename)
        s_multiindex = pd.MultiIndex.from_frame(s_df)
        s_multiindex.name = 's'
        s = xr.DataArray(s_multiindex)
        X['s'] = s
    
    var = list(X.keys())[0]
    return X[var]

def load_flattened_targets(path, datatype):
    """
    Loads flattened targets. Features should contain one multiindex:
    'sample', which is composed of times and ensemble member numbers.
    Surprise! The targets aren't actually flattened---only the sample 
    dimension is stacked. This function was named amidst a brain fart.
    
    Parameters:
        path: str
            Path to features.
        datatype: str
            dataname of the form 'train', 'val', or 'test'
    Returns:
        X: xr.DataArray
            The flattened targets with correct indexes.
    """
    y = xr.open_dataset(os.path.join(path, f'y_{datatype}.nc'))
    
    sample_filename = os.path.join(path, f'y_{datatype}_sample_coord.csv')
    if os.path.exists(sample_filename):
        sample_df = pd.read_csv(sample_filename)
        sample_multiindex = pd.MultiIndex.from_frame(sample_df)
        sample_multiindex.name = 'sample'
        sample = xr.DataArray(sample_multiindex)
        y['sample'] = sample
    
    #$ Return DataArray
    var = list(y.keys())[0]
    return y[var]

def load_flattened_dp_features(path, datatype):
    """
    Loads flattened DP features. Features should contain one multiindex:
    'sample', which is composed of times and ensemble member numbers.
    Surprise! The targets aren't actually flattened---only the sample 
    dimension is stacked. Brain fart again. Big blast!
    
    Parameters:
        path: str
            Path to features.
        datatype: str
            dataname of the form 'train', 'val', or 'test'
    Returns:
        X: xr.DataArray
            The flattened targets with correct indexes.
    """
    assert datatype in ['train', 'val', 'test']
    X_dp = xr.open_dataset(os.path.join(path, f'X_dp_{datatype}.nc'))
    
    # Get sample coordinate from target data
    sample_filename = os.path.join(path, f'y_{datatype}_sample_coord.csv')
    if os.path.exists(sample_filename):
        sample_df = pd.read_csv(sample_filename)
        sample_multiindex = pd.MultiIndex.from_frame(sample_df)
        sample_multiindex.name = 'sample'
        sample = xr.DataArray(sample_multiindex)
        X_dp['sample'] = sample
    
    #$ Return DataArray
    var = list(X_dp.keys())[0]
    return X_dp[var]