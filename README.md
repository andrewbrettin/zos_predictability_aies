# Daily-to-seasonal dynamic sea level predictabilty
This is the project repository for the paper "Uncertainty-permitting machine learning reveals sources of dynamic sea level predictability across daily-to-seasonal timescales", submitted to [Artificial Intelligence for the Earth Systems](https://www.ametsoc.org/index.cfm/ams/publications/journals/artificial-intelligence-for-the-earth-systems/).

<!-- Preprint:  -->

## Directory structure:
* `environment.yml`  Conda environment used for data processing and visualization.
* `jobqueue.yaml`  Configuration file for distributed computing.
* `paths.json`  Dictionary of directory paths.
* `globals.json`  Various global variables used in this project (constants, ensemble members, etc).
* `utils`  Utilities package. Inside there are modules related to data loading, routing data processing, computation, metrics, networks, plotting, xai, and clustering. This is installed to our environment in editable mode using `pip install -e .`.
* `data_processing/` Files listed are in chronological order.
  - `make_grids.py` Makes regridder file and computes areas. Saved to `scratch/grid/`.
  - `regrid.py` Regrids ocean variables and saves to `scratch/regridded/`.
  - `compute_sea_level_vars.ipynb` Computes inverse barometer contribution and effective sea level. Variables are saved to `scratch/sea_level/`.
  - `rechunk.ipynb` Rechunks variables to be optimized for detrending and deseasonalizing. Saved to `scratch/rechunked/`.
  - `detrend_deseasonalize.ipynb` Detrends and deseasonalizes variables. Saved to `scratch/detrended_deseasonalized`.
  - `rechunk_anom.ipynb` Rechunks deseasonalized data to be optimized for reading spatial fields. Saved to `scratch/anom_spatial/`.
  - `coarsen.py` Coarsens anomalies to a specified resolution. Output regridder files are saved to `coarsened/<resolution>/grid/` and anomalies are saved
  - `make_tensors.py`
  - `make_dp_tensors.py`
  - `save_icoords.py`
* `batch_scripts/` Batch scripts used for submission.
* `training/` Scripts for training networks and baselines.

## Conda environments
The following commands were used in the project repository to make a data analysis environment:

```bash
module load conda
conda create -n s2s python=3.8
conda activate s2s

conda install -c conda-forge xesmf 
conda install -c conda-forge dask netCDF4
pip install pytest
pytest -v --pyargs xesmf  # should all pass

module load cuda/11.7
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge numpy scipy pandas xarray
conda install -c conda-forge ipykernel ipywidgets tqdm
conda install -c conda-forge dask distributed dask-jobqueue joblib cython bottleneck
conda install -c conda-forge pytorch-lightning wandb
conda install -c conda-forge zarr cftime nc-time-axis
conda install -c conda-forge xrft xbatcher rechunker
conda install -c conda-forge -c pyviz matplotlib seaborn cartopy cmocean bokeh hvplot

conda update --all

pip install -e .

```

## Data overview
* The data is from the CESM2 large ensemble project
* Forcing: historical from 1850-2014, SSP370 from 2015-2100. Smoothed biomass-burning scenario.
* Use ensemble members 011, 012, and 013 from model initialization years of 1251, 1281, and 1301 (total of 9 ensemble members).
We don't use the macro states from ensembles initialized by years 1231, because UBOT and VBOT are missing the data from the historical SSP. 

In train.py, ensemble members 1281.013 and 1301.013 are used for validation and testing, respectively. All other ensemble members are used for training.

### Variables
We are given the variable `SSH_2`, which is the dynamic sea level (CMIP variable `sea_surface_height_above_geoid`). Due to the regridding, there is a small, insignificant global mean component to sea level at each time (varying on the order of ~<0.1mm each day). We remove this spatial mean and convert to meters and call this variable `zos`.

List of CESM2 variables:
* `SSH_2`: Output variable "sea-surface height". Corresponds to dynamic sea level, but has a nonzero spatial mean which must be removed for consistency with CMIP conventions. \[cm\].
* `SST`: Sea surface temperatures \[deg C\]
* `UBOT`: surface zonal wind speeds \[m s^-1\].
* `VBOT`: surface meridional wind speeds \[m s^-1\].
* `zos`: Dynamic sea level \[m\].


## Miscellaneous notes
* By convention, dates of tensor data correspond to the dates that a prediction is made. That is, `xr.DataArray.shift()` is applied to target tensors, not the input tensors.