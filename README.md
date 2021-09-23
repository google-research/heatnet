# HeatNet

HeatNet is a python package that provides tools to build, train and evaluate
neural networks designed to predict extreme heat wave events globally on daily
to subseasonal timescales. It also includes preprocessing tools for atmospheric
reanalysis data from the
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/#!/home).

## Dependencies

HeatNet relies on the [DLWP-CS](https://github.com/jweyn/DLWP-CS) project,
described in [Weyn et al. (2020)](https://doi.org/10.1029/2020MS002109), and
inherits all of its dependencies.

HeatNet requires installation of

- [TensorFlow](https://www.tensorflow.org) >= 2.0, to build neural networks
and data generators.
- [netCDF4](https://unidata.github.io/netcdf4-python/), to read and write
netCDF4 datasets.
- [xarray](http://xarray.pydata.org/en/stable/), to seamlessly manipulate
datasets and data arrays.
- [dask](http://xarray.pydata.org/en/stable/), to support parallel xarray
computations and streaming computation on datasets that don't fit into memory.
- [h5netcdf](https://anaconda.org/conda-forge/h5netcdf), which provides a
flexible engine for xarray I/O operations.
- [NumPy](https://numpy.org/install/) for efficient array manipulation.
- [cdsapi](https://cds.climate.copernicus.eu/api-how-to), to enable downloading
data from the Copernicus Climate Data Store.
- [TempestRemap](https://github.com/ClimateGlobalChange/tempestremap), for
mapping functions from latitude-longitude grids to cubed-sphere grids.

## Modules

- **data**: Classes and methods to download, preprocess and generate reanalysis
data for model training.
- **model**: Model architectures, custom losses and model estimators with
descriptive metadata.
- **eval**: Methods to evaluate model predictions, and compare against
persistence or climatology.
- **test**: Unit tests for classes and methods in the package.

## License

HeatNet is distributed under the GNU General Public License Version 3, which
means that any software modifying or relying on the HeatNet package must be
distributed under the same license. Consult the full notice to understand your
rights.

## Installation guide

The installation of heatnet and its dependencies has been tested with the
following configuration on both Linux and Mac personal workstations:

- Create a new Python 3.7 environment using [conda]
(https://www.anaconda.com/products/individual).
- In the terminal, activate the environment,  
`conda activate <environment_name>`.

- Install TensorFlow v2.3,  
`pip install tensorflow==2.3`
- Install xarray,  
`pip install xarray`
- Install netCDF4,  
`conda install netCDF4`
- Install TempestRemap,  
`conda install -c conda-forge tempest-remap`
- Install h5netcdf,  
`conda install -c conda-forge h5netcdf`
- Install pygrib (Optional),  
`pip install pygrib`
- Install cdsapi,  
`pip install cdsapi`
- Install h5py v2.10.0,  
`pip install h5py==2.10.0`
- Finally, install dask,  
`pip install dask`
- The DLWP package is not currently published, so the source code must be
downloaded from its GitHub repository. It is recommended to download this package in the same parent directory as HeatNet,  
`git clone https://github.com/jweyn/DLWP-CS.git`
- If you want to plot results using [Basemap](https://matplotlib.org/basemap/),
which is a slightly fragile (and deprecated) package, the following
configuration is compatible with this setup:  
`conda install basemap`  
`pip install -U matplotlib==3.2`

## Disclaimers
This is not an officially supported Google Product.
