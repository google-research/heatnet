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
"""Utils for raw and preprocessed ERA5 datasets."""
import os
from typing import List, Tuple

from absl import logging
import DLWP.model.preprocessing as dlwpp
import heatnet.data.cds_era5 as cds
import numpy as np
import xarray as xr


def get_varlev_pairs(variables: List[str], levels: List[float]) -> List[str]:
  """Returns a list of variable/level pairs from the input variables and levels.

  The length of variables and levels must be equal.

  Args:
    variables: List of non-unique variables.
    levels: List of non-unique levels.

  Returns:
    var_lev: List of variable/level pairs.
  """
  if len(variables) != len(levels):
    raise ValueError('For pairwise variable/level pairs, len(variables)='
                     f'{len(variables)} must equal len(levels)={len(levels)}.')
  var_lev = ['/'.join([v, str(int(l))]) for v, l in zip(variables, levels)]
  return var_lev


def coord_and_prefix(var_type: str) -> Tuple[str, str]:
  """Returns the appropriate coordinate and prefix for each variable type."""
  if var_type == 'predictors':
    coord_name = 'pred_varlev'
    da_prefix = 'pred'
  elif var_type == 'targets':
    coord_name = 'tgt_varlev'
    da_prefix = 'tgt'
  else:
    raise ValueError(f'var_type must be predictors or targets, not {var_type}.')
  return coord_name, da_prefix


def variable_moments(
    ds: xr.Dataset,
    variables: List[str],
    levels: List[float],
    batch_samples: int,
    var_type: str = 'predictors',
) -> Tuple[xr.DataArray, xr.DataArray]:
  """Returns DataArrays with the mean and std of all specified variables.

  The resulting DataArrays have metadata compatible with preprocessed data
  used to train HeatNet models.

  Args:
    ds: Dataset containing variables from which statistical moments are
      computed.
    variables: List of variables from which statistical moments are computed.
    levels: List of integer pressure levels (mb).
    batch_samples: Number of samples in the time dimension to read and process
      at once.
    var_type: Type of variables considered (predictors/targets).

  Returns:
    da_mean: A DataArray of mean values of variable/level pairs.
    da_std: A DataArray of standard deviations of variable/level pairs.
  """
  var_lev = get_varlev_pairs(variables, levels)
  surface_vars = cds.get_surface_vars(variables, formatting='short')
  n_var = len(var_lev)
  # Arrays for scaling parameters
  means = np.zeros((n_var,), dtype=np.float32)
  stds = np.ones((n_var,), dtype=np.float32)

  # Fill in the data. Iterate through variable/level pairs for scaling.
  for vl_index, (vl_var, vl_level) in enumerate(list(zip(variables, levels))):
    sel_kw = {} if (vl_var in surface_vars) else {'level': vl_level}

    logging.info(
        'Calculating mean and std of variable/level pair %s / %s'
        '(%s)', vl_index + 1, n_var, var_lev[vl_index])

    means[vl_index] = 1. * dlwpp.mean_by_batch(ds[vl_var].sel(**sel_kw),
                                               batch_samples)
    stds[vl_index] = 1. * dlwpp.std_by_batch(
        ds[vl_var].sel(**sel_kw), batch_samples, mean=means[vl_index])

  coord_name, da_prefix = coord_and_prefix(var_type)

  da_mean = xr.DataArray(
      means,
      coords={coord_name: var_lev},
      dims=[coord_name],
      name=da_prefix + '_mean')
  da_std = xr.DataArray(
      stds,
      coords={coord_name: var_lev},
      dims=[coord_name],
      name=da_prefix + '_std')
  return da_mean, da_std


def reduce_constant_dataset(ds_path: str,
                            ds_reduced_path: str,
                            overwrite: bool = False):
  """Reduces the size of a dataset with constant values in time.

  Takes a netCDF dataset with variables repeated along the time axis and writes
  another dataset to file omitting values along said axis.

  Args:
    ds_path: Path to netCDF Dataset.
    ds_reduced_path: Path to use when writing reduced netCDF Dataset.
    overwrite: If True, overwrites file in ds_reduced_path if it already exists.
  """
  if os.path.isfile(ds_reduced_path) and not overwrite:
    logging.warn('Output file %s already exists', ds_reduced_path)
    return

  with xr.open_dataset(ds_path) as ds:
    ds_reduced = ds.isel(time=[0]).squeeze().drop('time')
    if not ds_reduced.equals(ds.isel(time=[-1]).squeeze().drop('time')):
      raise ValueError('Dataset is not constant along time axis.')
    else:
      for key in ds_reduced.keys():
        ds_reduced = ds_reduced.assign(var_new=ds_reduced[key].astype(
            dtype=np.float32))
        ds_reduced = ds_reduced.drop_vars(key)
        ds_reduced = ds_reduced.rename({'var_new': key})
      logging.info('Writing reduced Dataset to file...')
      ds_reduced.to_netcdf(ds_reduced_path)
      ds_reduced.close()


def shard_ncdataset(orig_file_path: str,
                    new_file_path: str,
                    samples_per_shard: int = 128,
                    write_remainder: bool = False):
  """Shards a Dataset and saves shards to memory as netCDF files.

  Args:
    orig_file_path: Path to original Dataset to be sharded.
    new_file_path: Path for new Datasets, to which an integer will be appended,
      with .nc extension.
    samples_per_shard: Number of samples to store per shard, preferably power of
      2.
    write_remainder: Whether to write the remainder of samples to file.
  """
  with xr.open_dataset(orig_file_path) as full_ds:
    n_sample = full_ds.dims['sample']
    num_full_shards = n_sample // samples_per_shard
    new_file_ext = new_file_path.split('.')[-1]
    new_file_base = '.'.join(new_file_path.split('.')[:-1])
    for shard in range(num_full_shards):
      ds = full_ds.isel(
          sample=range(shard * samples_per_shard, (shard + 1) *
                       samples_per_shard))
      ds.to_netcdf('.'.join(
          (new_file_base, '0' * (6 - len(str(shard))) + str(shard),
           new_file_ext)))
    if write_remainder:
      ds = full_ds.isel(
          sample=range(num_full_shards * samples_per_shard, n_sample))
      ds.to_netcdf('.'.join(
          (new_file_base,
           '0' * (6 - len(str(num_full_shards))) + str(num_full_shards),
           new_file_ext)))
