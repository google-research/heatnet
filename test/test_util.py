# Copyright 2021 Google LLC
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
"""Utils for testing."""
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr


def write_dummy_dataset(path: str,
                        varname: str,
                        date_range: Tuple[str,
                                          str] = ('2016-12-31', '2017-12-31')):
  """Write dummy dataset with similar fields and attributes as ERA5 datasets.

  Args:
    path: Output path of written dataset.
    varname: Name of variable included in the dataset.
    date_range: Tuple specifying start and end dates for the dataset.
  """
  times = pd.date_range(date_range[0], date_range[1], freq='1D')
  lat = -np.arange(-90, 90)
  lon = np.arange(360)
  # Random data with random mean and standard deviation
  data = np.add(
      np.multiply(np.random.rand(),
                  np.random.rand(len(times), len(lat), len(lon))),
      np.random.rand())
  da = xr.DataArray(
      data,
      coords=([times, lat, lon]),
      dims=['time', 'latitude', 'longitude'],
      name=varname)
  da.to_dataset().to_netcdf(path)


def write_cubed_sphere_dataset(path: str,
                               varname: str,
                               date_range: Tuple[str, str] = ('2016-12-31',
                                                              '2017-12-31'),
                               samples_per_file: int = 1):
  """Write dummy dataset similar to processed data on the cubed sphere.

  Args:
    path: Output path of written dataset.
    varname: Name of variable included in the dataset.
    date_range: Tuple specifying start and end dates for the dataset.
    samples_per_file: Number of samples per file to use for storing the dataset.
  """
  assert path[-3:] == '.nc'
  times = pd.date_range(date_range[0], date_range[1], freq='1D')
  face = np.arange(6)
  height = np.arange(48)
  width = np.arange(48)
  # Random data with random mean and standard deviation.
  data_p = np.add(
      np.multiply(
          np.random.rand(),
          np.random.rand(len(times), 1, len(face), len(height), len(width))),
      np.random.rand())
  data_t = np.add(
      np.multiply(
          np.random.rand(),
          np.random.rand(len(times), 1, len(face), len(height), len(width))),
      np.random.rand())
  # Generate dummy predictors, targets, latitude and longitude fields.
  da_p = xr.DataArray(
      data_p,
      coords=([times, [varname], face, height, width]),
      dims=['sample', 'pred_varlev', 'face', 'height', 'width'],
      name='predictors')
  da_t = xr.DataArray(
      data_t,
      coords=([times, [varname], face, height, width]),
      dims=['sample', 'tgt_varlev', 'face', 'height', 'width'],
      name='targets')
  da_lat = xr.DataArray(
      data_p[0, 0],
      coords=([face, height, width]),
      dims=['face', 'height', 'width'],
      name='lat')
  da_lon = xr.DataArray(
      data_p[0, 0],
      coords=([face, height, width]),
      dims=['face', 'height', 'width'],
      name='lon')
  # Merge into dataset and shard.
  ds = xr.merge([da_p, da_t, da_lat, da_lon])
  num_files = len(times) // samples_per_file
  batches = np.split(times[:len(times) // num_files * num_files], num_files)
  for i, batch in enumerate(batches):
    # Files stored with .nc extension
    batch_path = '.'.join(
        (path[:-3], '0' * (6 - len(str(i))) + str(i), path[-2:]))
    ds.sel(sample=batch).to_netcdf(batch_path)
