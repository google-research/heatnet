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
"""File utils for external and local paths."""
import contextlib
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Generator, Union

from absl import logging
import tensorflow as tf
import xarray as xr


def ext_to_local(filename: str,
                 loc_dir: str,
                 overwrite: bool = False,
                 verbose: bool = False) -> str:
  """Copies an external file to a local directory.

  Args:
    filename: Path to file in external directory.
    loc_dir: Path to local directory where file will be copied.
    overwrite: Whether to overwrite the file if it exists at loc_dir.
    verbose: Whether to log progress statements.

  Returns:
    The local path where the file is written to.
  """
  loc_filename = os.path.join(loc_dir, os.path.basename(filename))
  maybe_make_dirs(loc_filename)
  if not os.path.exists(loc_filename) or overwrite:
    tf.io.gfile.copy(filename, loc_filename, overwrite=overwrite)
    if verbose:
      logging.info('Copied %s to local.', os.path.basename(loc_filename))
  else:
    if verbose:
      logging.info('File %s exists, omitting copy.',
                   os.path.basename(loc_filename))
  return loc_filename


def load_dataset(filename: str,
                 mode: str = 'local',
                 loc_dir: str = '/tmp/netcdf/data',
                 engine: str = 'h5netcdf',
                 **dataset_kwargs) -> xr.Dataset:
  """Loads a dataset from an external file, maybe copying it to local first.

  Args:
    filename: Path to file to load.
    mode: Method used to load data.
      'local': Copies files to a local path without deletion, and loads from
        local. This mode requires sufficient local storage to copy the whole
        dataset, and avoids copying data after the initial epoch.
      'tmp': Copies to a temporary local directory, and deletes files after
        loading. Only requires storage to load a copy of the current batch.
      'ext': Loads from the source directory directly. This may take longer than
        copying to local depending on the source path.
    loc_dir: Path to local directory where file will be copied, if mode is 'tmp'
      or 'local'.
    engine: Engine to use for xr.open_dataset(). Defaults to 'h5netcdf'.
    **dataset_kwargs: Arguments passed to xr.open_dataset().

  Returns:
    An xarray.Dataset.

  Raises:
    ValueError: If given mode is not one of ['ext', 'tmp', 'local'].
  """
  if mode == 'ext':
    with tf.io.gfile.GFile(filename, 'rb') as f:
      open_ds = xr.open_dataset(f, engine=engine, **dataset_kwargs)

  elif mode == 'tmp' or mode == 'local':
    loc_filename = os.path.join(loc_dir, os.path.basename(filename))
    ext_to_local(filename, loc_dir)
    open_ds = xr.open_dataset(loc_filename, engine=engine, **dataset_kwargs)
    if mode == 'tmp':
      tf.io.gfile.remove(loc_filename)

  else:
    raise ValueError(f'mode must be ext, tmp or local, but {mode} was passed.')

  return open_ds


@contextlib.contextmanager
def mkdtemp(**keyword_params) -> Generator[str, None, None]:
  """Create a local directory, removing it when the operation is complete.

  Args:
    **keyword_params: keyword params to be passed to tempfile.mkdtemp().

  Yields:
    Filename of the temporary file.
  """
  local_dirname = tempfile.mkdtemp(**keyword_params)
  try:
    yield local_dirname
  finally:
    shutil.rmtree(local_dirname)


@contextlib.contextmanager
def mktemp(**kwargs) -> Generator[str, None, None]:
  """Creates a local file, removing it when the operation is complete.

  Args:
    **kwargs: keyword parameters passed to tempfile.mkstemp().

  Yields:
    Filename of the temporary file.
  """
  fd, local_filename = tempfile.mkstemp(**kwargs)
  os.close(fd)
  try:
    yield local_filename
  finally:
    os.remove(local_filename)


def maybe_make_dirs(path: str) -> str:
  """Creates the subdirectories leading up to the given file, if nonexistent.

  Args:
    path: The path to a file (i.e., the last word in the path is assumed to be a
      file, not a directory).

  Returns:
    The generated path.
  """
  dirname = os.path.dirname(path)
  if dirname and not tf.io.gfile.exists(dirname):
    tf.io.gfile.makedirs(dirname)
  return dirname


def save_dict_to_json(dictionary: Dict[Any, Any], path: str):
  """Saves the given dictionary as a JSON file."""
  json_str = json.dumps(dictionary, indent=2) + '\n'
  maybe_make_dirs(path)
  with tf.io.gfile.GFile(path, 'w') as f:
    f.write(json_str)


def copy_dir(src_dir: str, dest_dir: str, **kwargs):
  """Recursively copy the src_dir to dist_dir.

  Args:
    src_dir: Path of the source directory.
    dest_dir: Path of the new directory.
    **kwargs: Keyword arguments passed to tf.io.gfile.copy().
  """
  for subdir_name, _, subdir_files in tf.io.gfile.walk(src_dir):
    new_subdir_name = os.path.join(dest_dir,
                                   os.path.relpath(subdir_name, src_dir))
    for subdir_file in subdir_files:
      new_file_path = os.path.join(new_subdir_name, subdir_file)
      maybe_make_dirs(new_file_path)
      tf.io.gfile.copy(
          os.path.join(subdir_name, subdir_file), new_file_path, **kwargs)


def save_netcdf(dataset: Union[xr.Dataset, xr.DataArray],
                path: str,
                netcdf_format: str = 'NETCDF4',
                overwrite: bool = False):
  """Writes dataset as netCDF4 file to external or local paths.

  Args:
    dataset: Dataset to write to file in netCDF format.
    path: Output path of written file.
    netcdf_format: Formatting protocol used when writing the dataset to file.
    overwrite: Whether to overwrite existing files or not.
  """
  with mkdtemp() as tmp_dir:
    loc_filename = os.path.join(tmp_dir, os.path.basename(path))
    dataset.to_netcdf(loc_filename, format=netcdf_format)
    if not tf.io.gfile.exists(path) or overwrite:
      tf.io.gfile.copy(loc_filename, path, overwrite=overwrite)
    else:
      logging.info('File %s already exists, omitting.', path)
    tf.io.gfile.remove(loc_filename)


def load_generic_netcdf(netcdf_path: str, **dataset_kwargs) -> xr.Dataset:
  """Loads a dataset from a generic netCDF file.

  Byte-like netCDF objects may be read by the scipy engine, if they are
  formatted as netCDF3, or by the h5netcdf engine if they are formatted as
  netCDF4. If byte-like object reading is not allowed by the OS, resort to
  reading from path.

  Args:
    netcdf_path: Path to the netCDF dataset.
    **dataset_kwargs: Keyword arguments passed to xr.open_dataset().

  Returns:
    The loaded dataset.
  """
  try:
    # Try netCDF3
    with tf.io.gfile.GFile(netcdf_path, 'rb') as f:
      return xr.open_dataset(f, engine='scipy', **dataset_kwargs)
  except TypeError:
    # Try netCDF4
    with tf.io.gfile.GFile(netcdf_path, 'rb') as f:
      return xr.open_dataset(f, engine='h5netcdf', **dataset_kwargs)
  except OSError:
    return xr.open_dataset(netcdf_path, **dataset_kwargs)
