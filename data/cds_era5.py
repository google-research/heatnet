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
"""Tools for fetching and processing data from Copernicus Climate Data Store."""
import copy
import os
from typing import List, Union

from absl import logging
import DLWP.data.era5
import heatnet.file_util as file_util
import xarray as xr

pressure_variable_names = {
    'divergence': 'd',
    'fraction_of_cloud_cover': 'cc',
    'geopotential': 'z',
    'ozone_mass_mixing_ratio': 'o3',
    'potential_vorticity': 'pv',
    'relative_humidity': 'r',
    'specific_cloud_ice_water_content': 'ciwc',
    'specific_cloud_liquid_water_content': 'clwc',
    'specific_humidity': 'q',
    'specific_rain_water_content': 'crwc',
    'specific_snow_water_content': 'cswc',
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'vertical_velocity': 'w',
    'vorticity': 'vo',
    'streamfunction': 'sf',
    'velocity_potential': 'vp'
}

surface_variable_names = {
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    '2m_dewpoint_temperature': 'd2m',
    '2m_temperature': 't2m',
    'land_sea_mask': 'lsm',
    'mean_sea_level_pressure': 'msl',
    'geopotential': 'z',
    'sea_surface_temperature': 'sst',
    'surface_latent_heat_flux': 'slhf',
    'surface_sensible_heat_flux': 'sshf',
    'surface_pressure': 'sp',
    'total_column_water': 'tcw',
    'total_column_water_vapour': 'tcwv',
    'total_precipitation': 'tp',
    'volumetric_soil_water_layer_1': 'swvl1',
    'volumetric_soil_water_layer_2': 'swvl2',
    'volumetric_soil_water_layer_3': 'swvl3',
    'volumetric_soil_water_layer_4': 'swvl4',
    'mean_top_net_long_wave_radiation_flux': 'mtnlwrf',
}


def get_short_name(variables: Union[str, List[str]]) -> Union[str, List[str]]:
  """Return the short name of long-name variables.

  Args:
    variables: Names of variables to be shortened.

  Returns:
    The shortened variable names.
  """
  all_variable_names = copy.deepcopy(pressure_variable_names)
  all_variable_names.update(surface_variable_names)
  if isinstance(variables, str):
    return all_variable_names[variables]
  else:
    return [all_variable_names[v] for v in variables]


def get_surface_vars(variables: List[str],
                     formatting: str = 'short') -> List[str]:
  """Returns all surface variables from an input variable list.

  Args:
    variables: Variable or list of variables in short or long name format.
    formatting: Whether the given variables are in 'short' or 'long' ERA5
      format.

  Returns:
    surface_vars: List of variables with no level coordinate.
  """
  if formatting == 'short':
    return [
        v for v in variables if v not in list(pressure_variable_names.values())
    ]
  elif formatting == 'long':
    return [
        v for v in variables if v not in list(pressure_variable_names.keys())
    ]
  else:
    raise ValueError(f'Format must be short or long, {formatting} was given.')


class CDSHandler(DLWP.data.era5.ERA5Reanalysis):
  """Handler for CDS data downloading and preprocessing."""

  def open(self, **dataset_kwargs):
    """Sets self.Dataset using the content of all self.raw_files."""
    if not self.dataset_variables:
      raise ValueError('The dataset_variables must be set before self.raw_files'
                       'are accessed.')
    self._set_file_names()
    if isinstance(self.raw_files, List):
      self.Dataset = xr.merge([
          file_util.load_generic_netcdf(file, **dataset_kwargs)
          for file in self.raw_files
      ])
    else:
      self.Dataset = file_util.load_generic_netcdf(self.raw_files,
                                                   **dataset_kwargs)
    self.dataset_dates = self.Dataset['time']

  def set_variables(self, variables: List[str]):
    """Set the variables to retrieve or open with the CDSHandler.

    Args:
      variables: Names of variables to download or open by the CDSHandler.
    """
    for v in variables:
      assert (v in list(pressure_variable_names.keys()) or v in list(
          surface_variable_names.keys())), (
              'Variables must be either in the pressure-level variables for'
              f'the dataset {list(pressure_variable_names.keys())}, or the'
              f'single-level variables {list(surface_variable_names.keys())}.')
    self.dataset_variables = sorted(variables)

  def get_climatology_anomalies(self,
                                variables: List[str],
                                levels: List[int],
                                standardize: bool = True,
                                store_climatology: bool = True,
                                include_original: bool = True,
                                verbose: bool = True,
                                overwrite: bool = False,
                                **dataset_kwargs):
    """Writes netCDF files for the climatology anomalies of 'variables'.

    Args:
      variables: Variables to subtract climatology from and create a new
        dataset.
      levels: List of integer pressure levels (mb).
      standardize: If True, divide anomalies by standard deviation.
      store_climatology: If True, writes climatology to file.
      include_original: If False, substitutes original variables by the
        anomalies in self.dataset_variables. If True, include both.
      verbose: If True, writes information about the function call.
      overwrite: If True, overwrites existing anomaly files.
      **dataset_kwargs: keyword arguments passed to open_dataset().
    """
    if not self.dataset_variables:
      self.set_variables(variables)
    elif not set(variables).issubset(self.dataset_variables):
      raise ValueError(f'All selected variable {variables} not included in'
                       f'{self.dataset_variables}.')
    self._set_file_names()
    var_levs = []
    for v in variables:
      if v not in list(pressure_variable_names.keys()) and v in list(
          surface_variable_names.keys()):
        var_levs.append((v, v))
      else:
        var_levs.extend([(v, v + '_' + str(level)) for level in levels])

    for variable, var_lev in var_levs:
      # self.raw_files have a .nc file extension
      for filename in self.raw_files:
        base_filename = var_lev + '.nc'
        if base_filename in filename:
          short_var = get_short_name(variable)
          var_file = filename
          proc_var_file = filename[:-3] + '_anom' + filename[-3:]
          if store_climatology:
            climat_file = filename[:-3] + '_climat' + filename[-3:]
          break

      self.dataset_variables.append(var_lev + '_anom')
      if not include_original and variable in self.dataset_variables:
        self.dataset_variables.remove(variable)

      if not overwrite and os.path.exists(proc_var_file):
        logging.warn('File %s already exists; omitting...', proc_var_file)
        continue

      if verbose:
        logging.info('Computing climatology of %s...', var_lev)

      raw_var = file_util.load_generic_netcdf(var_file, **dataset_kwargs)

      climat_mean = raw_var.groupby('time.dayofyear').mean('time')
      if standardize:
        climat_std = raw_var.groupby('time.dayofyear').std('time')
        var_anom = xr.apply_ufunc(lambda ds, m, s: (ds - m) / s,
                                  raw_var.groupby('time.dayofyear'),
                                  climat_mean, climat_std)
        if store_climatology:
          climat_mean = climat_mean.rename_vars(
              {short_var: '_'.join((
                  short_var,
                  'clim',
                  'mean',
              ))})
          climat_std = climat_std.rename_vars(
              {short_var: '_'.join((
                  short_var,
                  'clim',
                  'std',
              ))})
          climat_ds = xr.merge([climat_mean, climat_std], join='exact')
          file_util.save_netcdf(climat_ds, climat_file, overwrite=overwrite)
      else:
        var_anom = raw_var.groupby('time.dayofyear') - climat_mean
        if store_climatology:
          climat_mean = climat_mean.rename_vars(
              {short_var: '_'.join((
                  short_var,
                  'clim',
                  'mean',
              ))})
        file_util.save_netcdf(climat_mean, climat_file, overwrite=overwrite)

      var_anom = var_anom.rename_vars(
          {short_var: '_'.join((short_var, 'anom'))})
      file_util.save_netcdf(
          var_anom.drop('dayofyear'), proc_var_file, overwrite=overwrite)
      if verbose:
        logging.info('Climatological anomaly of %s written to file %s.',
                     var_lev, proc_var_file)

  def resample_files(self,
                     frequency: str,
                     verbose: bool = False,
                     overwrite: bool = False,
                     **dataset_kwargs):
    """Resamples self.raw_files and writes to netCDF files.

    The paths in self.raw_files are modified to include the resampled
    dataset paths, instead of the original dataset paths. The new paths
    are overriden by any call to set_filenames().

    Args:
      frequency: Resampling frequency following xarray notation.
      verbose: If True, writes information about the function call.
      overwrite: If True, overwrites existing anomaly files.
      **dataset_kwargs: keyword arguments passed to open_dataset().
    """
    self._set_file_names()
    new_raw_files = []
    for filename in self.raw_files:
      resampled_filename = '.'.join((filename[:-3], 'rsmp', filename[-2:]))
      new_raw_files.append(resampled_filename)
      if not overwrite and os.path.exists(resampled_filename):
        logging.warn('File %s already exists; omitting...', resampled_filename)
        continue
      if verbose:
        logging.info('Resampling file %s...', filename)

      ds_raw = file_util.load_generic_netcdf(filename, **dataset_kwargs)

      ds_resampled = ds_raw.resample(time=frequency).mean()
      file_util.save_netcdf(
          ds_resampled, resampled_filename, overwrite=overwrite)

    self.raw_files = new_raw_files
