"""
computing.py

Various computational utilities that are useful in the postprocessing
and/or analysis stage.
"""

__author__ = "@andrewbrettin"
__all__ = [
    "pearson_corr",
    "autocorr",
    "compute_lagged_correlations",
    "get_confident_samples",
    "get_composite",
    "compute_pval",
    "compute_significance"
]

import os
import sys
import numpy as np
import xarray as xr
import scipy as sc
import joblib
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_jobqueue import PBSCluster
from tqdm import tqdm
import xrft


def pearson_corr(x, y):
    """
    Computes Pearson correlation coefficients between two
    xarray DataArrays x and y.
    """
    # Broadcast dimensions
    if len(x.dims) < len(y.dims):
        x = x.broadcast_like(y)
    elif len(x.dims) > len(y.dims):
        y = y.broadcast_like(x)
    
    x_anom = x - x.mean(dim='time')
    y_anom = y - y.mean(dim='time')
    
    cov = (x_anom * y_anom).sum(dim='time')
    var_x = np.sqrt((x_anom**2).sum(dim='time')) 
    var_y = np.sqrt((y_anom**2).sum(dim='time')) 
    return cov / (var_x * var_y)

def autocorr(x, lag):
    y = x.shift(time=lag).isel(time=slice(lag, None))
    x = x.sel(time=y.time)
    return pearson_corr(x, y)

def compute_lagged_correlations(x, lags=30, verbose=10):
    """
    Computes autocorrelations at multiple time lags using multiple cores.
    """
    parallel = joblib.Parallel(n_jobs=-1, verbose=verbose)
    
    if isinstance(lags, int):
        lags = np.arange(0, lags)
    
    corrs = parallel([
        joblib.delayed(autocorr)(x, i) for i in lags
    ])
    
    return np.array(corrs)

def get_confident_samples(darray, pred_stds, conf_threshold=0.01, 
                          most_confident=True, sortby='confidence'):
    """
    Returns the samples yielding the most confident (or least confident)
    predictions.
    
    Parameters:
        darray: xr.DataArray
            array to get confident samples from
        pred_stds: array-like
            Predicted standard deviations corresponding to the times
            from the test set.
        pred_dates: xr.DataArray
            The dates giving the initial conditions for the forecast.
        conf_threshold: float
            Confidence percentile threshold. For instance, if
            conf_threshold = 0.2, we select the 20% most (or least)
            confident samples.
        most_confident: bool
            If True, return the samples corresponding to the most
            confident predictions. If False, return the samples for
            the least confident predictions
        sortby: str, 'confidence' or 'std' or 'time'
            Indicates whether samples are sorted by ascending confidence
            (i.e., decreasing std), ascending std, or chronological order.
    Returns
        conf_samples: xr.DataArray
    """
    # Get std indices sorted in decreasing order
    pred_std_array = np.array(pred_stds)
    sort_indices = np.argsort(pred_std_array)[::-1]
    
    N = len(pred_std_array)
    n = int(N * conf_threshold)
    
    if most_confident:
        conf_indices = sort_indices[-n:None]
    else:
        conf_indices = sort_indices[0:n]
    
    # Get times
    pred_dates = pred_stds['time']
    conf_times = pred_dates.isel(time=conf_indices)
    conf_samples = darray.sel(time=conf_times)
    
    # Rearrange values by sortby
    if sortby == 'confidence':
        pass
    elif sortby == 'std':
        conf_samples = conf_samples.sel(time=conf_samples['time'][::-1])
    elif sortby == 'time':
        conf_samples = conf_samples.sortby('time')
    else:
        raise ValueError('sortby must be either "confidence", "std", or "time"')
    
    return conf_samples


def get_composite(darray, pred_stds, prediction_dates, conf_threshold=0.01, most_confident=True):
    conf_samples = get_confident_samples(
        darray, pred_stds, prediction_dates, 
        conf_threshold=conf_threshold, most_confident=most_confident
    )
    
    composite = conf_samples.mean(dim='time')
    composite.attrs = conf_samples.attrs
    return composite

def compute_pval(darray, darray_conf, method='mannwhitneyu'):
    """
    Given climatological samples darray and a subset of samples
    darray_conf, computes a p-value for a two-sided z-test or
    Mann-Whitney U-test to determine whether the subset is distinct 
    from climatology in a statistically significant way.
    
    Parameters:
        darray: xr.DataArray
            Climatological values.
        darray_conf: xr.DataArray
            Subset of samples yielding confident (or uncertain)
            predictions.
        method: str
            If 'z_test', computes p-values according to a 2-sided 
            z-test. If 'mannwhitneyu', computes p-values according to
            a 2-sided Mann-Whitney U-test.
    Returns:
        p: xr.DataArray
            p-values for the null hypothesis at each lat-lon point.
    """
    if method == 'mannwhitneyu':
        U, p = xr.apply_ufunc(
            sc.stats.mannwhitneyu,
            darray, 
            darray_conf,
            input_core_dims=[['time', 'lat', 'lon'], ['time', 'lat', 'lon']],
            output_core_dims=[('lat', 'lon'), ('lat', 'lon')],
            exclude_dims={'time', 'lat', 'lon'},
            join='inner',
            kwargs={'axis': 0, 'nan_policy': 'propagate'},
            dask='allowed'
        )
        # Resolve issue with coordinates
        coords = darray.isel(time=0).drop_vars('time').coords
        p = p.assign_coords(coords)
        return p
    elif method == 'z_test':
        n = len(darray_conf['time'])
        Z_score = (
            darray_conf.mean(dim='time') / (darray.std(dim='time') / np.sqrt(n))
        )
        p = xr.apply_ufunc(
            lambda z: 2*(1-sc.stats.norm.cdf(np.abs(z))),
            Z_score,
            dask='allowed'
        )
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    p.name = 'p'
    p.attrs['long_name'] = 'p-value'
    p.attrs['units'] = ''
    return p

def compute_significance(darray, darray_conf, alpha=0.05, method='mannwhitneyu'):
    """
    Given climatological samples darray and a subset of samples
    darray_conf, applies a z-test or a Mann-Whitney U-test to 
    determine whether the subset is distinct from climatology
    in a statistically significant way.
    
    Parameters:
        darray: xr.DataArray
            Climatological values.
        darray_conf: xr.DataArray
            Subset of samples yielding confident (or uncertain)
            predictions.
        alpha: float
            Probability threshold for defining statistically
            significant samples.
        method: str
            If 'z_test', computes p-values according to a 2-sided 
            z-test. If 'mannwhitneyu', computes p-values according to
            a 2-sided Mann-Whitney U-test.
    Returns:
        sigs: xr.DataArray
            DataArray of booleans expressing whether the samples in
            darray_conf are distinct from climatological values in
            a statistically significant way.
    """
    p = compute_pval(darray, darray_conf, method=method)
    sigs = p < alpha
    
    # Mask nans
    sigs = xr.where(np.isnan(darray.isel(time=0)), np.nan, sigs)
    
    return sigs