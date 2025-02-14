"""
plotting.py

Various plotting utilities.
"""

import numpy as np
import scipy as sc
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cmocean as cmo
import nc_time_axis

import utils

from sklearn.linear_model import LinearRegression

#$ Globals
plot_colors = {
    'ssh': 'tab:blue',
    'ann': 'tab:orange',
    'dp': 'k'
}

#$ Utility functions
def override_kwargs(kwarg_defaults, kwarg_overrides):
    """
    Utility function for overriding keyworded arguments.
    
    Parameters:
        kwarg_defaults: dict
            Default dictionary of keyword arguments
        kwarg_overrides: dict
            Keyword arguments to override.
        
    Returns:
        kwargs:
            Dictionary of keyword arguments, with default values 
            overriden by override values.
            
    Example usage:
        kwargs = {}
        ax_kwargs = {'title': 'My truly special plot'}
    
        fig, ax = plt.subplots()
        ax.plot(x, y, **kwargs)
        
        ax_defaults = {'xlabel': 'x', 'title': 'My plot'}
        ax_kwargs = override_kwargs(ax_defaults, ax_kwargs)
        ax.set(**ax_kwargs)
    """
    kwargs = kwarg_defaults
    for k, v in kwarg_overrides.items():
        kwargs[k] = v
    return kwargs

def set_symmetric_ylim(ax):
    ymax = np.max(np.abs(np.array(ax.get_ylim())))
    ax.set(ylim=(-ymax, ymax))
    return ax

def calibration_plot(pred_mean_z, pred_std_z, target_z, res=0.05, 
                     confidence=0.95, ax=None, ax_kwargs={}, **kwargs):
    
    # Compute Z-scores
    p_list = np.arange(res, 1, res)
    Z_p = sc.stats.norm.ppf(p_list)
    Z_data = (target_z - pred_mean_z) / pred_std_z

    #$$ Point estimates of CDF
    fracs = np.empty_like(Z_p)
    for i, p in enumerate(p_list):
        fracs[i] = (Z_data <= Z_p[i]).mean()

    #$$ Confidence intervals (Dvoretzky-Kiefer-Wolfowitz theorem)
    conf_interval = np.empty((2, len(p_list)))
    N = len(target_z)
    alpha = 1 - confidence
    eps = np.sqrt(np.log(2/alpha) / (2*N))
    Z_p_lower = sc.stats.norm.ppf(p_list - eps)
    Z_p_upper = sc.stats.norm.ppf(p_list + eps)

    for i, p in enumerate(p_list):
        conf_interval[0, i] = (Z_data <= Z_p_lower[i]).mean()
        conf_interval[1, i] = (Z_data <= Z_p_upper[i]).mean()

    errors = np.abs(conf_interval - fracs)

    plt.style.use('default')
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
        
    default_kwargs = dict(ls='', marker='.', markersize=2, color='tab:red')
    kwargs = override_kwargs(default_kwargs, kwargs)
    ax.errorbar(p_list, fracs, yerr=errors, **kwargs)
    
    # Identity line
    ax.plot([0, 1], [0, 1], ls='--', lw=1, color='tab:grey', alpha=0.5)
    
    ax_defaults = dict(
        aspect='equal',
        facecolor=(0.95,0.95,0.95),
        title='Gaussianity of predictions',
        xlabel=r'$p$',
        ylabel=r'$\hat{F}_n(z_p)$',
        xlim=(0,1),
        ylim=(0,1),
        # facecolor='lightcyan'
    )
    
    ax_kwargs = override_kwargs(ax_defaults, ax_kwargs)
    ax.set(**ax_kwargs);
    
    fig.tight_layout()
    
    return ax