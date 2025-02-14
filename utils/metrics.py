"""
metrics.py

Various metrics 
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

#$ Pointwise performance metrics
def error(pred, true):
    """
    Pointwise error pred - true.
    """
    return pred - true

def abs_error(pred, true):
    """
    Absolute error |pred - true|.
    """
    return np.abs(true - pred)

def rel_error(pred, true, threshold=0.01):
    """
    Relative error (pred - true) / true. To prevent div by 0 
    denominator is capped at a threshold 0.01 m.
    """
    error = pred - true
    reldiff = error / np.maximum(np.abs(true), threshold)
    return reldiff

def rel_abs_error(pred, true, threshold=0.01):
    error = abs_error(true, pred)
    reldiff = error / np.maximum(np.abs(true), threshold)
    return reldiff

#$ Summary metrics
def correlation(pred, true):
    return np.corrcoef(pred, true)[0,1]

def MAE(pred, true):
    return np.abs(true - pred).mean()

def RMSE(pred, true):
    return np.sqrt(((true - pred)**2).mean())

def MARE(pred, true, threshold=0.01):
    absdiff = np.abs(true - pred)
    reldiff = absdiff / np.maximum(np.abs(true), threshold)
    return reldiff.mean()

def RMSRE(pred, true, threshold=0.01):
    reldiff = np.abs(true - pred) / np.maximum(np.abs(true), threshold)
    sre = reldiff**2
    msre = sre.mean()
    rmsre = np.sqrt(msre)
    return rmsre

def print_metrics(pred, true, model='', file=None):
    corr = correlation(pred, true).item()
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    mare = MARE(pred, true)
    rmsre = RMSRE(pred, true)
    
    print('{} corr:   {:4.2f}'.format(model, corr), file=file)
    print('{} MAE:    {:6.4f}'.format(model, mae), file=file)
    print('{} RMSE:   {:6.4f}'.format(model, rmse), file=file)
    print('{} MARE:   {:6.4f}'.format(model, mare), file=file)
    print('{} RMSRE:  {:6.4f}'.format(model, rmsre), file=file)

def metrics_df(pred, true, name=None):
    index = pd.Index(
        ['corr', 'MAE', 'RMSE', 'MARE', 'RMSRE'],
        name='metric'
    )
    
    if isinstance(pred, xr.DataArray):
        pred = pred.values
        true = true.values
        
    metrics_list = [
        correlation(pred, true),
        MAE(pred, true),
        RMSE(pred, true),
        MARE(pred, true),
        RMSRE(pred, true)
    ]
    
    metrics = pd.Series(metrics_list, index, name=name)
    return metrics