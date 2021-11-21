# Copied and modified from DLWP-CS/DLWP/remap
#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#
"""Tools for remapping on cubed-sphere coordinates from DLWP-CS.

Modified slightly to work inside Google's infrastructure.
"""
import os
import subprocess
from typing import Dict, Optional
import warnings

from absl import logging
import heatnet.file_util as file_util
import numpy as np
import pandas as pd
import xarray as xr

from google3.pyglib import resources

# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-return-or-yield


def to_chunked_dataset(ds, chunking):
  """Create a chunked copy of a Dataset with proper encoding for netCDF export.

  Args:
    ds: xarray.Dataset
    chunking: dict: chunking dictionary as passed to xarray.Dataset.chunk()

  Returns:
    ds: xarray.Dataset: chunked copy of ds with proper encoding
  """
  chunk_dict = dict(ds.dims)
  chunk_dict.update(chunking)
  ds_new = ds.chunk(chunk_dict)
  for var in ds_new.data_vars:
    ds_new[var].encoding['contiguous'] = False
    ds_new[var].encoding['original_shape'] = ds_new[var].shape
    ds_new[var].encoding['chunksizes'] = tuple(
        [c[0] for c in ds_new[var].chunks])
  return ds_new


class CubeSphereRemap(object):
  """Implements tools for remapping to and from a cubed sphere using TempestRemap executables."""

  def __init__(self,
               path_to_remapper: str,
               to_netcdf4: bool = True,
               verbose: bool = True):
    """Initializes a CubeSphereRemap object.

    Args:
      path_to_remapper: Path to the TempestRemap executables.
      to_netcdf4: if True, also use 'ncks' command to convert remapped files to
        netCDF4.
      verbose: print commands and progress.
    """

    # Modification to find the binary
    self.path_to_remapper = path_to_remapper
    logging.info('path to resources is %s ',
                 resources.GetARootDirWithAllResources())
    remapper_binary = os.path.join(resources.GetARootDirWithAllResources(),
                                   path_to_remapper)
    self.path_to_remapper = remapper_binary
    logging.info('path to binary is')
    logging.info(remapper_binary)

    self.remapper = os.path.join(remapper_binary, 'ApplyOfflineMap')
    self.map = None
    self.inverse_map = None
    self.to_netcdf4 = to_netcdf4
    self.verbose = verbose
    self._lat = None
    self._lon = None
    self._res = None
    self._map_exists = False
    self._inverse_map_exists = False

  def assign_maps(self,
                  map_name: Optional[str] = None,
                  inverse_map_name: Optional[str] = None):
    """Point to either or both of existing map conversion files for TempestRemap.

    Args:
      map_name: Path to forward remapping map.
      inverse_map_name:  Path to inverse remapping map.
    """
    if map_name is not None:
      self.map = map_name
      self._map_exists = True
    if inverse_map_name is not None:
      self.inverse_map = inverse_map_name
      self._inverse_map_exists = True

  def generate_offline_maps(self,
                            lat: int,
                            lon: int,
                            res: int,
                            map_name: Optional[str] = None,
                            inverse_map_name: Optional[str] = None,
                            inverse_lat: bool = False,
                            remove_meshes: bool = True,
                            in_np: int = 1,
                            output_dir: str = '/tmp/'):
    """Generate offline maps for cubed sphere remapping.

    Args:
      lat: Number of points in the latitude dimension
      lon: Number of points in the longitude dimension
      res: Number of points on a side of each cube face
      map_name: File name of the forward map
      inverse_map_name: File name of the inverse map
      inverse_lat: If True, then the latitudes in the data file are
        monotonically decreasing
      remove_meshes: If True, remove the temporary meshes generated while making
        the offline maps
      in_np: Order of transformation. Should be int in range 1 to 4.
      output_dir: output directory needed for running inside Google
    """
    assert lat > 0
    assert lon > 0
    assert res > 0
    assert 1 <= in_np <= 4
    self._lat = lat
    self._lon = lon
    self._res = res
    if map_name is None:
      self.map = 'map_LL%dx%d_CS%d.nc' % (self._lat, self._lon, self._res)
    else:
      self.map = map_name
    if inverse_map_name is None:
      self.inverse_map = 'map_CS%d_LL%dx%d.nc' % (self._res, self._lat,
                                                  self._lon)
    else:
      self.inverse_map = None

    self.map = os.path.join(output_dir, self.map)
    self.inverse_map = os.path.join(output_dir, self.inverse_map)

    logging.info('Self.map is at path: %s ', self.map)
    logging.info('CubeSphereRemap: generating offline forward map...')
    try:
      cmd = [
          os.path.join(self.path_to_remapper, 'GenerateRLLMesh'), '--lat',
          str(self._lat), '--lon',
          str(self._lon), '--file',
          os.path.join(output_dir, 'outLL.g')
      ]
      if inverse_lat:
        cmd = cmd + ['--lat_begin', '90', '--lat_end', '-90']
      if self.to_netcdf4:
        cmd = cmd + ['--out_format', 'Netcdf4']

      logging.info(' '.join(cmd))
      subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      logging.info('An error occurred while generating the lat-lon mesh.')
      logging.info(e.output)
      raise

    try:
      cmd = [
          os.path.join(self.path_to_remapper, 'GenerateCSMesh'), '--res',
          str(self._res), '--file',
          os.path.join(output_dir, 'outCS.g')
      ]
      if self.to_netcdf4:
        cmd = cmd + ['--out_format', 'Netcdf4']

      logging.info(' '.join(cmd))
      subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      logging.info('An error occurred while generating the cube sphere mesh.')
      logging.info(e.output)
      raise
    try:
      cmd = [
          os.path.join(self.path_to_remapper, 'GenerateOverlapMesh'), '--a',
          os.path.join(output_dir, 'outLL.g'), '--b',
          os.path.join(output_dir, 'outCS.g'), '--out',
          os.path.join(output_dir, 'ov_LL_CS.g')
      ]
      if self.to_netcdf4:
        cmd = cmd + ['--out_format', 'Netcdf4']

      logging.info(' '.join(cmd))
      subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      logging.info('An error occurred while generating the overlap mesh.')
      logging.info(e.output)
      raise
    try:
      cmd = [
          os.path.join(self.path_to_remapper,
                       'GenerateOfflineMap'), '--in_mesh',
          os.path.join(output_dir, 'outLL.g'), '--out_mesh',
          os.path.join(output_dir, 'outCS.g'), '--ov_mesh',
          os.path.join(output_dir, 'ov_LL_CS.g'), '--in_np',
          str(in_np), '--in_type', 'FV', '--out_type', 'FV', '--out_map',
          self.map
      ]
      if self.to_netcdf4:
        cmd = cmd + ['--out_format', 'Netcdf4']

      logging.info(' '.join(cmd))
      subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      logging.info('An error occurred while generating the offline map.')
      logging.info(e.output)
      raise
    self._map_exists = True

    logging.info('CubeSphereRemap: generating offline inverse map...')
    try:
      cmd = [
          os.path.join(self.path_to_remapper, 'GenerateOverlapMesh'), '--a',
          os.path.join(output_dir, 'outCS.g'), '--b',
          os.path.join(output_dir, 'outLL.g'), '--out',
          os.path.join(output_dir, 'ov_CS_LL.g')
      ]
      if self.to_netcdf4:
        cmd = cmd + ['--out_format', 'Netcdf4']

      logging.info(' '.join(cmd))
      subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      logging.info('An error occurred while generating the overlap mesh.')
      logging.info(e.output)
      raise
    try:
      cmd = [
          os.path.join(self.path_to_remapper,
                       'GenerateOfflineMap'), '--in_mesh',
          os.path.join(output_dir, 'outCS.g'), '--out_mesh',
          os.path.join(output_dir, 'outLL.g'), '--ov_mesh',
          os.path.join(output_dir, 'ov_CS_LL.g'), '--in_np',
          str(in_np), '--in_type', 'FV', '--out_type', 'FV', '--out_map',
          self.inverse_map
      ]
      if self.to_netcdf4:
        cmd = cmd + ['--out_format', 'Netcdf4']

      logging.info(' '.join(cmd))
      subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      logging.info('An error occurred while generating the offline map.')
      logging.info(e.output)
      raise

    if remove_meshes:
      for f in [
          os.path.join(output_dir, 'outLL.g'),
          os.path.join(output_dir, 'outCS.g'),
          os.path.join(output_dir, 'ov_LL_CS.g'),
          os.path.join(output_dir, 'ov_CS_LL.g')
      ]:
        os.remove(f)

    self._inverse_map_exists = True
    logging.info(
        'CubeSphereRemap: successfully generated offline maps (%s, %s)',
        self.map, self.inverse_map)

  def remap(self, input_file: str, output_file: str, *args):
    """Apply the forward remapping to an input_file, saved to output_file.

    Args:
      input_file: Path to input data file.
      output_file: Path to output data file.
      *args: Other arguments passed to the ApplyOfflineMap function

    Raises:
      ValueError: if the map is undefined
      FileNotFoundError: if the map is not found
    """
    if not self._map_exists:
      raise ValueError('No forward map has been defined or generated;'
                       "use 'generate_offline_maps' or "
                       "'assign_maps' functions first")
    elif not os.path.exists(self.map):
      raise FileNotFoundError(self.map)

    logging.info('CubeSphereRemap: applying forward map...')

    try:
      cmd = [
          os.path.join(self.path_to_remapper, 'ApplyOfflineMap'), '--in_data',
          input_file, '--out_data', output_file, '--map', self.map
      ] + list(args)
      logging.info(' '.join(cmd))
      subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      logging.info('An error occurred while applying the offline map.')
      logging.info(e.output)
      raise

    logging.info('CubeSphereRemap: successfully remapped data into %s',
                 output_file)

  def inverse_remap(self, input_file: str, output_file: str, *args):
    """Apply the forward remapping to data in an input_file.

    Args:
      input_file: Path to input data file.
      output_file: Path to output data file.
      *args: Other arguments passed to the ApplyOfflineMap function.

    Raises:
      ValueError: if the map is undefined
      FileNotFoundError: if the map is not found
    """
    if not self._inverse_map_exists:
      raise ValueError('No inverse map has been defined or generated; '
                       " use 'generate_offline_maps' or "
                       "'assign_maps' functions first")
    elif not os.path.exists(self.inverse_map):
      raise FileNotFoundError(self.inverse_map)

    logging.info('CubeSphereRemap: applying inverse map...')

    try:
      cmd = [
          os.path.join(self.path_to_remapper, 'ApplyOfflineMap'), '--in_data',
          input_file, '--out_data', output_file, '--map', self.inverse_map
      ] + list(args)
      logging.info(' '.join(cmd))
      subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      logging.info('An error occurred while applying the offline map.')
      logging.info(e.output)
      raise

    logging.info('CubeSphereRemap: successfully inverse remapped data into %s',
                 output_file)

  def convert_to_faces(self,
                       input_file: str,
                       output_file: str,
                       coord_file: Optional[str] = None,
                       chunking: Optional[Dict] = None) -> xr.Dataset:
    """Convert a dataset in cubed-sphere coordinates to contain (face, height, width) dimensions.

    Args:
      input_file: Input data file.
      output_file: Output data file.
      coord_file: If given, use this file to fill in missing coordinates that
        may have been removed by the remap() process.
      chunking: If provided, save the netCDF with this chunking ({dim:
        chunksize} pairs).

    Returns:
     The given dataset with new coordinates.
    """
    # Open the dataset to convert
    ds = file_util.load_generic_netcdf(input_file)
    logging.info('CubeSphereRemap.convert_to_faces: loading data to memory...')

    # First, assign any coordinates missing from the input file
    # from the coordinate file, if provided.
    if coord_file is not None:
      ds_coord = file_util.load_generic_netcdf(coord_file)
      missing_coordinates = [
          c for c in ds.dims.keys()
          if (c not in ds.coords.keys() and c != 'ncol')
      ]
      for coord in missing_coordinates:
        if coord not in ds_coord.coords.keys():
          warnings.warn("coordinate '%s' missing in coordinate file; omitting" %
                        coord)
          continue
        ds = ds.assign_coords(**{coord: ds_coord.coords[coord]})

    # Create a multi-index dimension
    n_width = int(np.sqrt(ds.dims['ncol'] // 6))
    face_index = pd.MultiIndex.from_product(
        (range(6), range(n_width), range(n_width)),
        names=('face', 'height', 'width'))

    # Assign the new coordinate and transpose
    new_dims = tuple([d for d in ds.dims.keys() if d != 'ncol'
                     ]) + ('face', 'height', 'width')
    logging.info(
        'CubeSphereRemap.convert_to_faces: assigning new coordinates to dataset'
    )
    ds_new = ds.assign_coords(ncol=face_index).unstack('ncol').transpose(
        *new_dims)

    # Export to a new file
    logging.info(
        'CubeSphereRemap.convert_to_faces: exporting data to file %s...',
        output_file)
    if chunking is not None:
      ds_new = to_chunked_dataset(ds_new, chunking)
    file_util.save_netcdf(ds_new, output_file)

    logging.info('convert_to_faces: successfully exported reformatted data')
    return ds_new

  def convert_from_faces(self,
                         input_file: str,
                         output_file: str,
                         chunking: Optional[Dict] = None) -> xr.Dataset:
    """Convert a dataset with (face, height, width) dimensions to remapper coordinates.

    The final coordinates are the default 'ncol' dimension, which can be inverse
    remapped
    by the remapper.

    Args:
      input_file: Input data file
      output_file: Output data file
      chunking: If provided, save the netCDF with this chunking ({dim:
        chunksize} pairs)

    Returns:
     The given dataset with new coordinates.
    """
    # Open the dataset to convert
    ds = file_util.load_generic_netcdf(input_file)
    logging.info(
        'CubeSphereRemap.convert_from_faces: loading data to memory...')

    # Transpose the face dimension and stack the face, height, width
    fhw = ('face', 'height', 'width')
    new_dims = tuple([d for d in ds.dims.keys() if d not in fhw]) + fhw
    logging.info('CubeSphereRemap: assigning new coordinates to dataset')
    ds_new = ds.transpose(*new_dims).stack(ncol=fhw).reset_index('ncol')

    # Export to new file
    logging.info(
        'CubeSphereRemap.convert_from_faces: exporting data to file %s...',
        output_file)
    if chunking is not None:
      ds_new = to_chunked_dataset(ds_new, chunking)
    file_util.save_netcdf(ds_new, output_file)

    logging.info(
        'CubeSphereRemap.convert_from_faces: successfully exported reformatted data'
    )
    return ds_new
