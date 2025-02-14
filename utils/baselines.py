"""
baselines.py

Contains classes for fitting baselines to training data
"""


__author__ = "@andrewbrettin"
__all__ = [
    "DampedPersistenceModel",
    "load_train_zos",
    "load_test_zos"
]

import os
import sys
import json
import itertools
import numpy as np
import scipy as sc
import xarray as xr
import utils
sys.path.append('..')

with open("~/s2s/paths.json") as paths_json:
    PATHS = json.load(paths_json)
with open("~/s2s/globals.json") as globals_json:
    GLOBALS = json.load(globals_json)


class DampedPersistenceModel():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        """
        Sets damping coefficient beta from arrays X and y.
        
        Parameters:
            X: array-like
                Training features. If a list of arrays, coefficients
                beta are computed for each list element and averaging.
            y: array-like
                Training targets.
        Returns: self
        """
        
        if isinstance(X, list) and isinstance(y, list):
            betas = []
            for (X_data, y_data) in zip(X, y):
                beta = compute_beta(X, y)
                betas.append(beta)
            self.beta = np.mean(betas)
            return self
        else:
            self.beta = compute_beta(X, y)
        
        return self
    
    def fit_experiment(self, experiment, tau, s=5):
        """
        Sets damping coefficient beta using the data for a given 
        experiment.
        
        Parameters:
            experiment: str
                Experiment name.
            tau: int
                Forecast lead time, in days.
            s: int
                Sampling frequency of the data.
        Returns: self
        
        Beta is fitted using experiment targets (5-day averaged zos
        anomalies at a given point) and targets shifted in time as
        features (e.g., 5-day averaged zos at an earlier time). 
        Therefore, the correlation coefficient beta is higher than if
        we were to compute it using daily averaged fields instead of
        5-day averaged fields.
        """
        configs = get_configs(experiment)
        data_path = os.path.join(PATHS['tensors'], configs['data'])
        
        #$ Load training data
        y_train = utils.data.load_flattened_targets(data_path, 'train')
        
        betas = []
        for member_id in np.unique(y_train['member_id']):
            #$ Select specific member id
            y_train_member = y_train.sel(member_id=member_id)

            #$ Infer features from targets by shifting values backward
            #$ tau days (i.e. tau//s indices)
            X_train_member = y_train_member.shift(time=tau//s)

            #$ Drop nan values
            X_train_member = X_train_member.dropna(dim='time')

            #$ Select same time
            y_train_member = y_train_member.sel(time=X_train_member.time)

            #$ Get arrays
            X = X_train_member.values
            y = y_train_member.values

            #$ Do regression
            beta = np.dot(X,y) / sc.linalg.norm(X)**2

            betas.append(beta)
        
        #$ Set beta to be average over all members
        self.beta = np.mean(betas)
        
        #$ Also, set tau and s to be attributes
        self.tau = tau
        self.s = s
        
        return self

    def fit_to_training_data(self, point, tau, T=5, s=5, quick_train=False):
        """
        This function determines the damping coefficient beta by 
        using the daily-averaged fields from the 7 ensemble members 
        used for training. We directly use the data from 
        utils.data.load_anomalies.
        
        Parameters:
            point: dict of 'lat', 'lon'
                Dictionary giving the spatial location of timeseries.
            tau: int
                Forecast lead time.
            T: int
                Averaging period for dynamic sea levels.
            s: int
                Sampling frequency in days. Default is 5 days, which
                is the same as the default averaging period for targets
                T=5.
            quick_train: boolean
                If true, only training member (1251, 011) is used.
                Otherwise all training members are used.
        Returns: self
        """
        #$ Training data
        if quick_train:
            train_members = [(1251, '011')]
        else:
            train_members = list(
                itertools.product(GLOBALS['init_years'], GLOBALS['members']))
            train_members.remove((1281, '013'))
            train_members.remove((1301, '013'))

        #$ Loop through train members and compute beta
        betas = []
        for init_year, member in train_members:
            #$ Load training data
            X, y = get_dp_data(init_year, member, point, tau=20, T=5, s=5)
            
            #$ Fit coefficients for ensemble member
            beta = compute_beta(X, y)
            betas.append(beta)
        
        #$ Compute composite beta
        self.beta = np.mean(betas)
        
        #$ Also, set tau and s to be attributes
        self.tau = tau
        self.T = T
        self.s = s
        
        return self
    
    def predict(self, X):
        """
        Makes prediction for standardized values X.
        """
        if isinstance(X, xr.DataArray):
            X = X.values
        
        pred = self.beta * X    
        return pred
    
def get_dp_data(init_year, member, point, tau=20, T=5, s=5, c=1):
    """
    This function gets the dp data from base information (ensemble 
    member, time lag, time averaging, sampling rate.
    
    Assumptions:
    * Input/output data has the same spatial resolution of 1deg.
    * Input/output data is averaged over the same period of T days.
    * Sampling rate of 1 sample every 5 days.
    
    Parameters:
        init_year: int
            Initialization year.
        member: str
            Ensemble member.
        tau: int
            Time lag.
        T: int or list of int
            Time averaging of inputs and outputs.
        s: int
            Sampling rate. should be 5 (for sampling every 5 days).
    
    """
    X_dp = utils.data.load_anomalies(
        'zos', init_year, member, chunkedby='time')

    X_dp = (
        X_dp
        .sel(point, method='nearest')
        .rolling(time=T, center=False).mean()   # Rolling means
        .isel(time=slice(T-1, None))            # Subselection
        .isel(time=slice(0, None, s))           # Subsample
    )
    
    y_dp = X_dp.shift(time=-tau//s)

    # Drop nans incurred from shifting
    y_dp = y_dp.isel(time=slice(0,-tau//s))
    X_dp = X_dp.sel(time=y_dp['time'])
    
    return X_dp, y_dp
    
def compute_beta(X, y):
    beta = np.dot(X, y) / sc.linalg.norm(X)**2
    return beta

