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
"""Processing tools for ingestion of CDS ERA5 data into HeatNet models."""
import datetime
import os
from typing import List, Tuple, Union, Optional

from absl import logging
import heatnet.data.cds_era5 as cds
import heatnet.file_util as file_util
import netCDF4 as nc
import numpy as np
import pandas as pd
import tensorflow as tf
from .util import get_varlev_pairs
from .util import variable_moments
import xarray as xr


class CDSPreprocessor(object):
  """A Preprocessor of CDS data to train HeatNet models."""

  def __init__(self,
               raw_files: Union[str, List[str]],
               predictor_vars: Union[None, str, List[str]] = None,
               target_vars: Union[None, str, List[str]] = None,
               predictor_levels: Union[None, float, List[float]] = None,
               target_levels: Union[None, float, List[float]] = None,
               lead_times: Union[None, List[int]] = None,
               past_times: Union[None, List[int]] = None,
               base_out_path: str = 'preproc_data.nc',
               write_mode: str = 'write',
               verbose=False,
               **dataset_kwargs):
    """Initialize a CDSPreprocessor for conversion of CDS data to train data.

    If given as lists, elements in variables and levels must have a one to
    one correspondence.

    Args:
      raw_files: File(s) containing raw input data.
      predictor_vars: List of predictor variables to include in processed file.
        If None, includes all variables in raw_files.
      target_vars: List of target variables to include in processed file. If
        None, includes all variables in raw_files.
      predictor_levels: List of integer pressure levels (mb) for predictors. If
        None, includes all levels in raw_files.
      target_levels: List of integer pressure levels (mb) for targets. If None,
        includes all levels in raw_files.
      lead_times: List of forecast lead times to include as targets. If None,
        only one lead time is chosen, given by the time_step.
      past_times: List of past times to include as predictors. If None, only use
        present data as predictors.
      base_out_path: Base file path of output files with .nc extension.
      write_mode: Writing mode for netCDF files. 'copy' writes to local and
        copies to a possibly external final path. Choose 'write' to write
        directly to the final path.
      verbose: If True, print progress statements.
      **dataset_kwargs: Arguments passed to file_util.load_dataset().
    """
    self.raw_files = raw_files
    self.open(**dataset_kwargs)
    self._dt = pd.to_timedelta(
        (self.dataset_dates[1] - self.dataset_dates[0]).values)
    self.base_out_path = base_out_path
    file_util.maybe_make_dirs(self.base_out_path)
    self.pred_vars, self.pred_levels = self.parse_variables_levels(
        predictor_vars, predictor_levels)
    self.tgt_vars, self.tgt_levels = self.parse_variables_levels(
        target_vars, target_levels)
    self.past_times = past_times if past_times is not None else [0]
    self.lead_times = lead_times if lead_times is not None else [1]
    self.write_mode = write_mode
    self.verbose = verbose

  def open(self, **dataset_kwargs):
    """Opens a multifile dataset combining the raw input files.

    Args:
      **dataset_kwargs: kwargs passed to file_util.load_dataset().
    """
    if isinstance(self.raw_files, List):
      self.raw_ds = xr.merge([
          file_util.load_dataset(file, **dataset_kwargs)
          for file in self.raw_files
      ])
    else:
      self.raw_ds = file_util.load_dataset(self.raw_files, **dataset_kwargs)
    self.dataset_dates = self.raw_ds['time']

  def close(self):
    """Closes the CDSPreprocessor.raw_ds."""
    self.raw_ds.close()

  def get_output_paths(self, n_sample: int, batch_samples: int) -> List[str]:
    """Returns a list of output paths given number of samples and batch size.

    Args:
      n_sample: The ceiling of number of samples to be written to file. The
        actual number is lower if the remainder with respect to batch_samples is
        not zero.
      batch_samples: The number of samples to batch in a single output file.

    Returns:
      The list of output paths.
    """
    # Drop remainder data by default
    num_out_paths = n_sample // batch_samples
    if num_out_paths * batch_samples < n_sample:
      logging.warning(
          'Dropping a remainder of %s samples from the processed data.',
          n_sample - num_out_paths * batch_samples)

    max_digits = 6 if num_out_paths < 1e6 else len(str(num_out_paths))
    # self.base_out_path has .nc extension
    return [
        '.'.join(
            (self.base_out_path[:-3], '0' * (max_digits - len(str(i))) + str(i),
             self.base_out_path[-2:])) for i in range(num_out_paths)
    ]

  def parse_variables_levels(
      self, variables: Union[None, str, List[str]],
      levels: Union[None, float, List[float]]) -> Tuple[List[str], List[float]]:
    """Parses a set of variables and levels into a pair of equal length lists.

    Args:
      variables: Variables from the raw dataset to include in the processed
        dataset. If None, includes all variables in dataset.
      levels: Levels from the raw dataset to include in the processed dataset.
        If None, includes all levels in dataset.

    Returns:
      parsed_vars: List of variables.
      parsed_levels: List of levels.

    Raises:
      ValueError: When only one of variables and levels are None, or when
       variables and levels are lists of different length.
    """
    if variables is None and levels is None:
      all_levels = []
      all_variables = []
      for varname in self.raw_ds.data_vars.keys():
        if 'level' not in self.raw_ds[varname].coords:
          all_levels.append(0.0)
          all_variables.append(varname)
        else:
          all_levels.extend(self.raw_ds[varname].coords['level'].values)
          all_variables.extend([varname] *
                               len(self.raw_ds[varname].coords['level'].values))
      parsed_vars = all_variables
      parsed_levels = all_levels
    elif variables is None or levels is None:
      raise ValueError('variables and levels must both be specified, or'
                       f'both set to None, but variables={variables} and'
                       f'levels={levels} were given.')
    else:
      parsed_vars = variables if isinstance(variables, List) else [variables]
      parsed_levels = levels if isinstance(levels, List) else [levels]

      if len(parsed_vars) != len(parsed_levels):
        raise ValueError(
            'For pairwise variable/level pairs, len(variables)='
            f'{len(parsed_vars)} must equal len(levels)={len(parsed_levels)}.')

    return parsed_vars, parsed_levels

  def get_varlev_time_triads(self,
                             var_lev: List[str],
                             times: Union[int, List[int]],
                             is_it_predictors: bool = True) -> List[str]:
    """Gets list of variable/level/time offset triads used as dataset channels.

    Args:
      var_lev: List of variable/level pairs.
      times: List of time offsets to include for each variable/level pair with
        respect to the 'time' coordinate of the processed dataset.
      is_it_predictors: If True, the variable/level pairs are predictors. Else,
        they are targets.

    Returns:
      var_lev_time: List of variable/level/time triads, which correspond to
        single channels in the processed dataset.
    """
    if not isinstance(times, List):
      times = [times]

    suffix = 'D' if self._dt == datetime.timedelta(days=1) else 'dt'
    var_lev_time = []
    for varlev_i in var_lev:
      if is_it_predictors:
        var_lev_time.append(varlev_i)
        var_lev_time.extend(
            [varlev_i + '/-' + str(time) + suffix for time in times if time])
      else:
        var_lev_time.extend(
            [varlev_i + '/+' + str(time) + suffix for time in times])
    return var_lev_time

  def get_mean_and_std_for_scale(self,
                                 overwrite: bool) -> Tuple[xr.DataArray, ...]:
    """Returns the mean and std of predictors and targets.

    If the statistics are computed, they are also written to file in netCDF
    format.

    Args:
      overwrite: If True, overwrites statistics found in file.

    Returns:
      pred_mean: DataArray containing mean of predictors.
      pred_std: DataArray containing standard deviation of predictors.
      tgt_mean: DataArray containing mean of targets.
      tgt_std: DataArray containing standard deviation of targets.
    """
    scale_file = '.'.join(
        (self.base_out_path[:-3], 'scales', self.base_out_path[-2:]))
    if tf.io.gfile.exists(scale_file) and not overwrite:
      logging.info('Using existing scale file %s for normalization.',
                   scale_file)
      with tf.io.gfile.GFile(scale_file, 'rb') as f:
        with xr.open_dataset(f, engine='h5netcdf') as scale_ds:
          return (scale_ds['pred_mean'], scale_ds['pred_std'],
                  scale_ds['tgt_mean'], scale_ds['tgt_std'])

    else:
      pred_mean, pred_std = variable_moments(
          self.raw_ds,
          self.pred_vars,
          self.pred_levels,
          4096,
          var_type='predictors')
      tgt_mean, tgt_std = variable_moments(
          self.raw_ds, self.tgt_vars, self.tgt_levels, 4096, var_type='targets')

      self.pred_varlev = get_varlev_pairs(self.pred_vars, self.pred_levels)
      self.tgt_varlev = get_varlev_pairs(self.tgt_vars, self.tgt_levels)
      if self.write_mode == 'copy':
        final_scale_file = scale_file
        scale_file = os.path.basename(scale_file)
      with nc.Dataset(scale_file, 'w') as nc_scale:
        nc_scale.description = 'Scaling parameters for data'
        nc_scale.createDimension('pred_varlev', len(self.pred_varlev))
        nc_scale.createDimension('tgt_varlev', len(self.tgt_varlev))
        nc_var = nc_scale.createVariable('pred_varlev', str, 'pred_varlev')
        nc_var.setncatts({
            'long_name': 'Predictor variable/level pair',
        })
        nc_scale.variables['pred_varlev'][:] = np.array(
            self.pred_varlev, dtype='object')

        nc_var = nc_scale.createVariable('tgt_varlev', str, 'tgt_varlev')
        nc_var.setncatts({
            'long_name': 'Target variable/level pair',
        })
        nc_scale.variables['tgt_varlev'][:] = np.array(
            self.tgt_varlev, dtype='object')

        # Create means and stds variables
        nc_var = nc_scale.createVariable('pred_mean', np.float32,
                                         ('pred_varlev',))
        nc_var.setncatts({
            'long_name': 'Global mean of variable/level predictor pairs',
            'units': 'N/A',
        })
        nc_var[:] = pred_mean.values

        nc_var = nc_scale.createVariable('pred_std', np.float32,
                                         ('pred_varlev',))
        nc_var.setncatts({
            'long_name':
                'Global std deviation of variable/level predictor pairs',
            'units':
                'N/A',
        })
        nc_var[:] = pred_std.values

        # Targets
        nc_var = nc_scale.createVariable('tgt_mean', np.float32,
                                         ('tgt_varlev',))
        nc_var.setncatts({
            'long_name': 'Global mean of variable/level target pairs',
            'units': 'N/A',
        })
        nc_var[:] = tgt_mean.values

        nc_var = nc_scale.createVariable('tgt_std', np.float32, ('tgt_varlev',))
        nc_var.setncatts({
            'long_name': 'Global std deviation of variable/level target pairs',
            'units': 'N/A',
        })
        nc_var[:] = tgt_std.values
      if self.write_mode == 'copy':
        tf.io.gfile.copy(scale_file, final_scale_file, overwrite=overwrite)
        tf.io.gfile.remove(scale_file)
      return pred_mean, pred_std, tgt_mean, tgt_std

  def raw_to_batched_samples(self,
                             batch_samples: int = 32,
                             scale_variables: bool = False,
                             chunk_size: int = 1,
                             overwrite: bool = False,
                             with_coord: bool = True):
    """Convert self.raw_ds to a Dataset of input/output batched samples.

    Args:
      batch_samples: Number of samples in the time dimension to read, process
        and write at once.
      scale_variables: If True, de-means and scales variables by standard
        deviation on a variable/level basis. The full dataset statistics are
        used, and written to file.
      chunk_size: Size of the chunks in the sample (time) dimension.
      overwrite: If True, overwrites any existing output files.
      with_coord: If True, the returned Dataset contains the variable/level
        pairs as coordinates.
    """
    if not with_coord:
      self.base_out_path = self.base_out_path + '.nocoord'

    self.pred_varlev = get_varlev_pairs(self.pred_vars, self.pred_levels)
    self.tgt_varlev = get_varlev_pairs(self.tgt_vars, self.tgt_levels)
    self.pred_varlev_time = self.get_varlev_time_triads(
        self.pred_varlev, self.past_times, is_it_predictors=True)
    self.tgt_varlev_time = self.get_varlev_time_triads(
        self.tgt_varlev, self.lead_times, is_it_predictors=False)
    if self.verbose:
      logging.info('Predictor variable/level pairs: %s', self.pred_varlev_time)
      logging.info('Target variable/level pairs: %s', self.tgt_varlev_time)
    # Reserve times for offsets.
    first_time = max(self.past_times)
    n_sample = len(self.dataset_dates) - (first_time) - max(self.lead_times)
    out_paths = self.get_output_paths(n_sample, batch_samples)

    if scale_variables:
      pred_mean, pred_std, tgt_mean, tgt_std = self.get_mean_and_std_for_scale(
          overwrite=overwrite)

    else:
      pred_mean = None
      pred_std = None
      tgt_mean = None
      tgt_std = None

    lat_dim = 'lat' if 'lat' in self.raw_ds.dims.keys() else 'latitude'
    lon_dim = 'lon' if 'lon' in self.raw_ds.dims.keys() else 'longitude'
    n_lat, n_lon = (self.raw_ds.dims[lat_dim], self.raw_ds.dims[lon_dim])
    for file_id, out_path_i in enumerate(out_paths):
      if tf.io.gfile.exists(out_path_i) and not overwrite:
        if self.verbose:
          logging.info('Output file %s already exists, omitting.', out_path_i)
        continue
      if self.verbose:
        logging.info(
            'Processor.raw_to_batched_samples: creating output file'
            '%s/%s', out_path_i, out_paths)
      if self.write_mode == 'copy':
        final_out_path_i = out_path_i
        out_path_i = os.path.basename(out_path_i)
      with nc.Dataset(out_path_i, 'w') as nc_fid:
        nc_fid.description = 'Training data for model'
        nc_fid.setncattr('scaling', 'True' if scale_variables else 'False')
        nc_fid.createDimension('sample', 0)
        nc_fid.createDimension('pred_varlev', len(self.pred_varlev_time))
        nc_fid.createDimension('tgt_varlev', len(self.tgt_varlev_time))
        nc_fid.createDimension('lat', n_lat)
        nc_fid.createDimension('lon', n_lon)

        # Create spatial coordinates
        nc_var = nc_fid.createVariable('lat', np.float32, 'lat')
        nc_var.setncatts({'long_name': 'Latitude', 'units': 'degrees_north'})
        nc_fid.variables['lat'][:] = self.raw_ds[lat_dim].values

        nc_var = nc_fid.createVariable('lon', np.float32, 'lon')
        nc_var.setncatts({'long_name': 'Longitude', 'units': 'degrees_east'})
        nc_fid.variables['lon'][:] = self.raw_ds[lon_dim].values

        if with_coord:
          nc_var = nc_fid.createVariable('pred_varlev', str, 'pred_varlev')
          nc_var.setncatts({
              'long_name': 'Predictor variable/level pair',
          })
          nc_fid.variables['pred_varlev'][:] = np.array(
              self.pred_varlev_time, dtype='object')

          nc_var = nc_fid.createVariable('tgt_varlev', str, 'tgt_varlev')
          nc_var.setncatts({
              'long_name': 'Target variable/level pair',
          })
          nc_fid.variables['tgt_varlev'][:] = np.array(
              self.tgt_varlev_time, dtype='object')

        # Create initialization time reference variable
        nc_var = nc_fid.createVariable('sample', np.float32, 'sample')
        time_units = 'hours since 1970-01-01 00:00:00'

        nc_var.setncatts({
            'long_name': 'Sample start time',
            'units': time_units
        })
        batch_init = first_time + batch_samples * file_id
        batch_end = min(first_time + batch_samples * (file_id + 1),
                        first_time + n_sample)
        times = np.array([
            datetime.datetime.utcfromtimestamp(d / 1e9)
            for d in self.raw_ds['time'].values[batch_init:batch_end].astype(
                datetime.datetime)
        ])
        nc_fid.variables['sample'][:] = nc.date2num(times, time_units)

        # Create predictors and targets variables
        pred_dims = ('sample', 'pred_varlev', 'lat', 'lon')
        target_dims = ('sample', 'tgt_varlev', 'lat', 'lon')
        chunks = (chunk_size, 1, n_lat, n_lon)

        predictors = nc_fid.createVariable(
            'predictors', np.float32, pred_dims, chunksizes=chunks)
        predictors.setncatts({
            'long_name': 'Predictors',
            'units': 'N/A',
            '_FillValue': np.array(nc.default_fillvals['f4']).astype(np.float32)
        })
        targets = nc_fid.createVariable(
            'targets', np.float32, target_dims, chunksizes=chunks)
        targets.setncatts({
            'long_name': 'Targets',
            'units': 'N/A',
            '_FillValue': np.array(nc.default_fillvals['f4']).astype(np.float32)
        })

        predictors = self.populate_varlev(
            predictors,
            n_sample,
            batch_samples,
            file_id,
            da_mean=pred_mean,
            da_std=pred_std)

        targets = self.populate_varlev(
            targets,
            n_sample,
            batch_samples,
            file_id,
            da_mean=tgt_mean,
            da_std=tgt_std)
      if self.write_mode == 'copy':
        tf.io.gfile.copy(out_path_i, final_out_path_i, overwrite=overwrite)
        tf.io.gfile.remove(out_path_i)

  def populate_varlev(self,
                      nc_variable: nc.Variable,
                      n_sample: int,
                      batch_samples: int,
                      nc_id: int,
                      da_mean: Optional[xr.DataArray] = None,
                      da_std: Optional[xr.DataArray] = None) -> nc.Variable:
    """Populate 'nc_variable' with training data aggregated from raw_ds.

    Data is from specified variables and levels contained in the
    'raw_ds'. If t_offsets is given, the returned variable contains the
    variable/level pairs at different time offsets as different channels. For
    predictors, offset times are interpreted as past times. For targets, as
    future times.

    Args:
      nc_variable: netCDF variable containing fields to be populated by method.
      n_sample: Total number of samples of each variable.
      batch_samples: Number of samples in the time dimension to read and process
        at once.
      nc_id: ID of the current batch to convert to ingestible netCDF.
      da_mean: DataArray containing the mean of variable/level pairs.
      da_std: DataArray containing the std of variable/level pairs.

    Returns:
      nc_variable: a netCDF variable containing arrays of variable/level/time
        triads for training.
    """
    # Define indexing functions from the raw_ds times given batch and offset.
    if nc_variable.name == 'predictors':
      variables = self.pred_vars
      levels = self.pred_levels
      t_offsets = self.past_times
      if 0 not in t_offsets:
        t_offsets.insert(0, 0)

      def idx_func_off(batch_init, t_offset):
        start = batch_init + max(t_offsets) - t_offset
        stop = min(start + batch_samples, n_sample + max(t_offsets) - t_offset)
        return slice(start, stop)

    elif nc_variable.name == 'targets':
      variables = self.tgt_vars
      levels = self.tgt_levels
      t_offsets = self.lead_times

      def idx_func_off(batch_init, t_offset):
        max_date_len = len(self.raw_ds['time'])
        past_times = max_date_len - max(t_offsets) - n_sample
        start = batch_init + past_times + t_offset
        stop = min(start + batch_samples, past_times + n_sample + t_offset)
        return slice(start, stop)

    var_lev = get_varlev_pairs(variables, levels)
    # Fill in the data. Iterate by variable and level for scaling.
    for vl_index, (vl_var, vl_level) in enumerate(list(zip(variables, levels))):
      if 'level' not in self.raw_ds[vl_var].coords:
        ds_vl = self.raw_ds[vl_var]
      else:
        ds_vl = self.raw_ds[vl_var].sel(level=vl_level)
      if self.verbose:
        logging.info('Processing variable/level pair %s of %s (%s) in %s.',
                     vl_index + 1, len(var_lev), var_lev[vl_index],
                     nc_variable.name)
      if da_mean is not None and da_std is not None:
        if nc_variable.name == 'predictors':
          v_mean = da_mean.sel(pred_varlev=var_lev[vl_index]).values
          v_std = da_std.sel(pred_varlev=var_lev[vl_index]).values
        elif nc_variable.name == 'targets':
          v_mean = da_mean.sel(tgt_varlev=var_lev[vl_index]).values
          v_std = da_std.sel(tgt_varlev=var_lev[vl_index]).values

      sample_init = nc_id * batch_samples
      for t_off_ind, t_off in enumerate(t_offsets):
        idx_orig = idx_func_off(sample_init, t_off)
        vlt_index = vl_index * len(t_offsets) + t_off_ind
        if da_mean is not None and da_std is not None:
          nc_variable[slice(batch_samples), vlt_index,
                      ...] = (ds_vl.isel(time=idx_orig).values - v_mean) / v_std
        else:
          nc_variable[slice(batch_samples), vlt_index,
                      ...] = ds_vl.isel(time=idx_orig).values

    return nc_variable
