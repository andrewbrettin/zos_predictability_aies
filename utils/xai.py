"""
xai.py

Various functions related to explainable AI tools and understanding predictability.
"""

__author__ = "@andrewbrettin"
__all__ = [
    "get_gradients",
    "inputs_times_gradients",
]

from tqdm import tqdm

import numpy as np
import xarray as xr
import torch


def get_gradients(model, inputs):
    """
    Computes the gradients for an input sample.
    
    Parameters:
        model: pl.LightningModule
            Network for computing gradients.
        inputs: torch.tensor
            NxD tensor of input samples, where N is the number of
            samples and D is the number of features.
    Returns:
        gradients: np.array
            Gradients
    """
    # Make sure inputs are correct
    inputs = inputs.clone().detach().requires_grad_(True)
    
    # Compute outputs
    preds = model(inputs)
    
    # Compute gradients
    shape_tensor = torch.ones_like(preds)
    gradients = torch.autograd.grad(preds, inputs, grad_outputs=shape_tensor)[0]
    
    # Detach
    gradients = gradients.detach()
    
    return gradients

def inputs_times_gradients(model, inputs):
    """
    Computes the gradients for an input sample.
    
    Parameters:
        model: pl.LightningModule
            Network for computing gradients.
        inputs: torch.tensor
            NxD tensor of input samples, where N is the number of
            samples and D is the number of features.
    Returns:
        products: np.array
            Inputs times gradients.
    """
    gradients = get_gradients(model, inputs)
    products = inputs.detach() * gradients
    
    return products

def get_integrated_gradients(model, inputs, baseline=None, num_steps=20):
    # Make sure inputs are ok
    inputs = inputs.clone().detach().requires_grad_(True)

    # Create zero baseline
    input_size = np.shape(inputs)[1:]
    if baseline is None:
        baseline = torch.zeros(input_size, dtype=torch.float32)
    baseline = torch.tensor(baseline, dtype=torch.float32)

    # Evaluate derivative at interpolants
    diff = (inputs - baseline)
    interpolator = torch.tensor(np.linspace(0, 1, num=num_steps))

    gradients = torch.zeros((num_steps, *inputs.shape))
    for i, frac in tqdm(enumerate(interpolator)):
        point = baseline + frac * diff
        gradients[i, :, :] = get_gradients(model, point)

    # Compute integrated gradients
    integrated_gradients = diff * gradients.mean(axis=0)

    integrated_gradients = integrated_gradients.detach().numpy()

    return integrated_gradients