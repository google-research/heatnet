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
"""Tensorflow generators for the HeatNet model."""
import collections.abc as abc
import concurrent.futures
import functools
import math
from typing import Dict, List, Optional, Tuple, Union

from absl import logging
import DLWP.util
import heatnet.file_util as file_util
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr


class BaseDataGenerator(tf.keras.utils.Sequence):
  """Base generator class for heatnet data.

  This class also makes it possible to add model-invariant data, such as
  incoming solar radiation or latitude, to the inputs.
  """

  def __init__(
      self,
      rank: int = 3,
      input_sel: Optional[Dict[str, str]] = None,
      output_sel: Optional[Dict[str, str]] = None,
      add_latlon: bool = False,
      topography_file: Optional[str] = None,
      lsm_file: Optional[str] = None,
      add_insolation: bool = False,
      add_time: bool = False,
      batch_size: int = 32,
      shuffle: bool = False,
      remove_nan: bool = True,
      channels_last: bool = True,
      drop_remainder: bool = True,
      verbose: bool = False,
  ):
    """Initializes a BaseDataGenerator.

    Args:
      rank: The number of spatial dimensions (e.g. 2 for 2-d data, 3 for data on
        the cubed sphere).
      input_sel: Variable/level selection for input features.
      output_sel: Variable/level selection for output features.
      add_latlon: If True, adds sin(latitude) and normalized longitude to the
        inputs.
      topography_file: If str, adds normalized orography loaded from the given
        str path.
      lsm_file: If str, adds land sea mask loaded from the given str path.
      add_insolation: If True, adds the daily max insolation without diurnal
        cycle (normalized).
      add_time: bool: if True: add normalized time (date) to the inputs.
      batch_size: Number of samples to take at a time from the dataset.
      shuffle: If True, randomly select batches.
      remove_nan: If True, remove any samples with NaNs.
      channels_last: If True, returns data with channels as the last dimension.
      drop_remainder: If True, ignores the last batch of data if it is smaller
        than the batch size.
      verbose: Controls logging of the Generator contruction process.
    """
    self.verbose = verbose
    self._add_latlon = add_latlon
    self._add_insolation = add_insolation
    self._add_time = add_time
    self._topo_file = topography_file
    self._lsm_file = lsm_file
    self._n_added_channels = (2 * int(self._add_latlon) + int(self._add_time) +
                              int(self._add_insolation))
    if self._topo_file is not None:
      self._n_added_channels += 1
    if self._lsm_file is not None:
      self._n_added_channels += 1
    self._batch_size = batch_size
    self._shuffle = shuffle
    self._remove_nan = remove_nan
    self._indices = []
    self.rank = rank
    self._input_sel = input_sel or {}
    self._output_sel = output_sel or {}
    self.drop_remainder = drop_remainder
    # Transpose option
    self.channels_last = channels_last
    self._transpose = (0,) + tuple(range(2, 2 + self.rank)) + (1,)
    # Placeholders for input and output shapes
    self.input_shape = (0,) * (self.rank + 1)
    self.output_shape = (0,) * (self.rank + 1)

  @property
  def n_features(self) -> int:
    """Returns the number of input features; including added channels."""
    return (np.prod(self.input_shape) +
            np.prod(self.input_shape[-self.rank:]) * self._n_added_channels)

  @property
  def convolution_shape(self) -> Tuple[int, ...]:
    """Returns the shape of inputs expected by convolutional layers.

    Shape (face, height, width, channels), including channels added during
    generation, for channels_last and rank=3.
    """
    conv_shape = (int(np.prod(self.input_shape[:-self.rank])) +
                  self._n_added_channels,) + self.input_shape[-self.rank:]
    if self.channels_last:
      return tuple([conv_shape[s - 1] for s in self._transpose[1:]])
    else:
      return conv_shape

  @property
  def output_n_features(self) -> int:
    """Returns the number of output features."""
    return np.prod(self.output_shape)

  @property
  def output_convolution_shape(self) -> Tuple[int, ...]:
    """Returns the shape of predictors expected by convolutional layers.

    Shape (face, height, width, channels), for channels_last and rank=3.
    """
    result = (np.prod(
        self.output_shape[:-self.rank]),) + self.input_shape[-self.rank:]
    if self.channels_last:
      return tuple([result[s - 1] for s in self._transpose[1:]])
    else:
      return result

  def __len__(self) -> int:
    """Returns the number of batches per epoch."""
    if self.drop_remainder:
      return self.n_sample // self._batch_size
    else:
      return math.ceil(self.n_sample / self._batch_size)

  def set_input_output_sel(self, ds: xr.Dataset):
    """Sets the selected inputs and outputs from the data for the generator.

    Args:
      ds: Dataset used to extract input and output selections.
    """
    if not self._input_sel:
      self._input_sel = {'pred_varlev': ds['pred_varlev'].values}
    if not self._output_sel:
      self._output_sel = {'tgt_varlev': ds['tgt_varlev'].values}

  def load_static_predictors(self, ds: xr.Dataset):
    """Loads all requested sample-independent static predictors to memory.

    If specified, loads the sin(latitude), the normalized longitude, the
    normalized topography, and the land-sea mask. These predictors are then
    added to the sampled data on the file during generate(). If no static
    predictors are added, self.static_pred is initialized as an empty list.

    Args:
      ds: Dataset used to extract latitude and longitude values.
    """
    self.static_pred = []
    self.lat = ds.lat.values
    self.lon = ds.lon.values
    if self._add_latlon:
      # sin(lat) and normalized lon
      lat_pred = np.repeat(
          np.sin((np.pi / 180.) * np.expand_dims(self.lat, axis=[0, 1])),
          self._batch_size,
          axis=0)
      lon_pred = np.repeat(
          np.divide(np.expand_dims(self.lon, axis=[0, 1]), 360.0),
          self._batch_size,
          axis=0)

      self.static_pred.append(lat_pred)
      self.static_pred.append(lon_pred)

    if self._topo_file is not None:
      # Normalized topography
      with tf.io.gfile.GFile(self._topo_file, 'rb') as f:
        with xr.open_dataset(f, engine='h5netcdf') as z_ds:
          z0 = z_ds.z.values
          z0_mean = np.mean(z0)
          z0_std = np.std(z0)
          topography_pred = np.repeat(
              np.divide(
                  np.subtract(np.expand_dims(z0, axis=[0, 1]), z0_mean),
                  z0_std),
              self._batch_size,
              axis=0)
          self.static_pred.append(topography_pred)
    if self._lsm_file is not None:
      with tf.io.gfile.GFile(self._lsm_file, 'rb') as f:
        with xr.open_dataset(f, engine='h5netcdf') as lsm_ds:
          lsm_pred = np.repeat(
              np.expand_dims(lsm_ds.lsm.values, axis=[0, 1]),
              self._batch_size,
              axis=0)
          self.static_pred.append(lsm_pred)
    if self.static_pred:
      self.static_pred = np.concatenate(self.static_pred, axis=1)

  def get_time_dep_predictors(self,
                              times: np.ndarray) -> List[Optional[np.ndarray]]:
    """Loads all requested time-dependent predictors to memory.

    Args:
      times: Array of times for which time-dependent predictors should be
        created.

    Returns:
      A list of requested time-dependent predictors, if any.
    """
    added_channels = []
    if self._add_insolation:
      insol = DLWP.util.insolation(times, self.lat, self.lon, daily=True)
      added_channels.append(insol[:, np.newaxis])

    if self._add_time:
      # time: days since 1970 Jan 1st, normalized by days from 1970 to 2020
      time_da = np.divide(
          np.expand_dims(
              times, axis=[1, 2, 3,
                           4]).astype('datetime64[D]').astype(dtype=np.float32),
          18250.0)
      time_da = np.repeat(
          time_da, self.input_shape[-self.rank], axis=-self.rank).repeat(
              self.input_shape[-self.rank + 1], axis=-self.rank + 1).repeat(
                  self.input_shape[-self.rank + 2], axis=-self.rank + 2)
      added_channels.append(time_da)

    return added_channels

  def to_tf_dataset(
      self,
      predictors: List[np.ndarray],
      targets: List[np.ndarray],
      batch_size: Optional[int] = None,
      input_names: Optional[List[str]] = None,
      output_names: Optional[List[str]] = None,
  ) -> tf.data.Dataset:
    """Obtains a tf.Dataset from a BaseDataGenerator.

    Args:
      predictors: A single sample of predictors (i.e., inputs).
      targets: A single sample of targets (i.e., outputs).
      batch_size: If int, use a fixed batch size. Will cause an error if the
        last batch of training data does not have the same number of samples.
      input_names: List of names for the inputs, to match the model Input layer.
      output_names: List of names for the outputs, to match the model output
        layer.

    Returns:
      A tf.data.Dataset.
    """
    if input_names is None:
      input_names = ['input_%d' % (i + 1) for i in range(len(predictors))]
    if len(input_names) != len(predictors):
      raise ValueError('Mismatched length of input names relative to'
                       f'generated data; got {len(input_names)} but expected'
                       f'{len(predictors)}.')

    if output_names is None:
      output_names = ['output'
                     ] + ['output_%d' % i for i in range(1, len(targets))]
    if len(output_names) != len(targets):
      raise ValueError('Mismatched length of output names relative to'
                       f'generated data; got {len(output_names)} but expected'
                       f'{len(targets)}.')

    def yield_fn():
      for sample in self:
        yield ({input_names[i]: d for i, d in enumerate(sample[0])},
               {output_names[i]: d for i, d in enumerate(sample[1])})

    data_types = ({input_names[i]: tf.float32 for i in range(len(predictors))},
                  {output_names[i]: tf.float32 for i in range(len(targets))})
    data_shapes = ({
        input_names[i]: (batch_size,) + predictors[i].shape[1:]
        for i in range(len(predictors))
    }, {
        output_names[i]: (batch_size,) + targets[i].shape[1:]
        for i in range(len(targets))
    })

    return tf.data.Dataset.from_generator(
        yield_fn, output_types=data_types, output_shapes=data_shapes)


class ShardedDataGenerator(BaseDataGenerator):
  """Generator of training data from a Dataset stored in multiple files.

  The ShardedDataGenerator may be used to train using batches
  spanning multiple files, or using batches pointing at single data files.

  The current implementation only allows training using data without remainders.
  In the case of batches spanning multiple files, only full files are used,
  meaning that samples from partially requested files are dropped from the
  generator. In the single file case, only batches fully contained in each file
  are read, meaning that the remainder samples are dropped from the generator.
  The Generator raises errors in these corner cases to prevent misuse.
  """

  def __init__(self,
               req_timestamps: List[pd.Timestamp],
               base_path: str,
               mode: str = 'local',
               **kwargs):
    """Initializes a ShardedDataGenerator that may use data from external paths.

    Args:
      req_timestamps: List of sample times to use in the generator.
      base_path: Path to input files, including base name.
      mode: Method used to load data during generate().
        'local': Copies files to a local path without deletion, and loads from
          local. This mode requires sufficient local storage to copy the whole
          dataset, and avoids copying data after the initial epoch.
        'tmp': Copies to a temporary local directory, and deletes files after
          loading. Only requires storage to load a copy of the current batch.
        'ext': Loads from the source directory directly. This may take longer
          than copying to local depending on the source path.
      **kwargs: BaseDataGenerator keyword arguments.
    """
    super(ShardedDataGenerator, self).__init__(drop_remainder=True, **kwargs)
    self.mode = mode
    self.base_path = base_path
    file_list = tf.io.gfile.glob(base_path + '*')
    file_list.sort()
    self.filenames = self.filelist_without_remainder(file_list, req_timestamps)
    num_files = len(self.filenames)
    self.set_input_output_sel()
    self.n_sample = num_files * self.samples_per_file
    self.batches_per_file = len(self) // num_files
    self.files_per_batch = num_files // len(self)
    # Initialize shuffling dictionaries
    self.batch_to_file = {}
    self.batch_to_samples = {}
    # Possibly shuffle dataset
    self.on_epoch_end()
    self.load_static_predictors()
    logging.info('Generator initialized.')

  def set_input_output_sel(self):
    """Sets the selected inputs and outputs from the data for the generator."""
    with tf.io.gfile.GFile(self.filenames[0], 'rb') as f:
      with xr.open_dataset(f, engine='h5netcdf') as ds:
        BaseDataGenerator.set_input_output_sel(self, ds)

  def filelist_without_remainder(
      self, file_list: List[str],
      req_timestamps: List[pd.Timestamp]) -> List[str]:
    """Returns the file list used by the generator, dropping any remainders.

    The returned file list drops a number of files such that the returned file
    list has no sample remainder with respect to batches or files. This function
    assumes that all data files in file_list contain the same number of equally
    spaced samples, except possibly the last file. For other sharding
    configurations, this function may not behave as intended.

    Args:
      file_list: The file list, possibly containing sample remainders.
      req_timestamps: List of requested samples. The returned samples are a
        subset of this list.

    Returns:
      The file list without sample remainders.

    Raises:
      NotImplementedError: Either the number of samples per file is divisible by
        the batch size or viceversa. Any other configuration raises an error.
      ValueError: If the requested data range contains samples outside the range
        of the data in file_list, or the last of files in the returned list has
        a different number of samples per file than the other files (i.e., it
        is a remainder file).
    """
    req_start = req_timestamps[0]
    req_end = req_timestamps[-1]
    with tf.io.gfile.GFile(file_list[0],
                           'rb') as f, tf.io.gfile.GFile(file_list[1],
                                                         'rb') as g:
      with xr.open_dataset(
          f, engine='h5netcdf') as ds, xr.open_dataset(
              g, engine='h5netcdf') as ds_2:

        self.samples_per_file = ds.dims['sample']
        first_ds_sample = ds.sample.values[0]
        first_file_ind = math.ceil((req_start - first_ds_sample) /
                                   (ds_2.sample.values[0] - first_ds_sample))

        if first_ds_sample > req_start:
          raise ValueError(
              f'Available data in provided path do not contain the'
              ' requested sample range. The requested start sample is'
              f' {req_start} and the first available time is'
              f' {first_ds_sample}.')

        if self.samples_per_file >= self._batch_size:
          logging.info('Using single file per batch mode.')
          if self.samples_per_file % self._batch_size > 0:
            raise NotImplementedError('The batch size must be a divisor of the'
                                      'number of samples per file in single'
                                      'file mode.')
          # Drop sample remainder
          last_file_ind = math.floor(
              (req_end - first_ds_sample) /
              (ds_2.sample.values[0] - first_ds_sample)) - 1
        else:
          logging.info('Using multiple files per batch mode.')
          if self._batch_size % self.samples_per_file > 0:
            raise NotImplementedError('The number of samples per file must be a'
                                      'divisor of the batch size in multifile'
                                      'mode.')
          # Drop file remainder with respect to batches
          num_req_files = math.floor(
              (req_end - req_start) / (ds_2.sample.values[0] - first_ds_sample))
          last_file_ind = first_file_ind + int(
              (num_req_files * self.samples_per_file) // self._batch_size *
              self._batch_size / self.samples_per_file) - 1

        if self.verbose:
          logging.info('Generator will use data from files %s to %s.',
                       first_file_ind, last_file_ind)
        if last_file_ind >= len(file_list):
          raise ValueError(
              f'Available data in provided path do not contain the'
              ' requested sample range. The requested end file has index'
              f' {last_file_ind} and the number of files is {len(file_list)}.')

    with tf.io.gfile.GFile(file_list[last_file_ind], 'rb') as f:
      with xr.open_dataset(f, engine='h5netcdf') as ds:
        if ds.dims['sample'] < self.samples_per_file:
          raise NotImplementedError(
              'Method only accepts shards with equal'
              'number of samples at the moment. Reset the requested sample set.'
          )
    return file_list[first_file_ind:last_file_ind + 1]

  def load_static_predictors(self):
    """Loads all requested sample-independent static predictors to memory.

    If specified, loads the sin(latitude), the normalized longitude, the
    normalized topography, and the land-sea mask. These predictors are then
    added to the sampled data on the file during generate(). If no static
    predictors are added, self.static_pred is initialized as an empty list.
    """
    self.static_pred = []
    with tf.io.gfile.GFile(self.filenames[0], 'rb') as f:
      with xr.open_dataset(f, engine='h5netcdf') as ds:
        self.input_shape = ds.predictors.isel(sample=[0]).sel(
            **self._input_sel).shape[1:]
        self.output_shape = ds.targets.isel(sample=[0]).sel(
            **self._output_sel).shape[1:]
        BaseDataGenerator.load_static_predictors(self, ds)

  def __getitem__(self,
                  index: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Returns one batch of data.

    Args:
      index: Index of requested batch.

    Returns:
      predictors: predictors array.
      targets: targets array.
    Raises:
      IndexError: if index > len(self).
    """
    # Generate indices of the batch
    if index < 0:
      index = len(self) + index
    elif index > len(self):
      raise IndexError

    batch_index = self._batch_indices[index]
    filename = self.batch_to_file[batch_index]
    samples = self.batch_to_samples[batch_index]

    return self.generate(samples, filename)

  def on_epoch_end(self):
    """Shuffles samples after each epoch if shuffling is requested.

    In single file per batch mode, shuffles the batches and
    within-file samples. In multifile per batch mode, shuffles
    batches and batch-to-file mappings.
    """
    self._batch_indices = np.arange(len(self))
    if self.batches_per_file > 0:
      sample_indices = np.arange(self.samples_per_file)
      if self._shuffle:
        np.random.shuffle(self._batch_indices)
        np.random.shuffle(sample_indices)
      for batch_id in np.arange(len(self)):
        # Associated files to each batch index.
        self.batch_to_file[batch_id] = self.filenames[math.floor(
            batch_id / self.batches_per_file)]
        # Associate within-file sample ranges to each batch index.
        self.batch_to_samples[batch_id] = sample_indices[
            self._batch_size *
            (batch_id % self.batches_per_file):self._batch_size *
            (batch_id % self.batches_per_file + 1)]
    else:
      sample_indices = np.arange(self._batch_size)
      if self._shuffle:
        np.random.shuffle(self._batch_indices)
        np.random.shuffle(self.filenames)
      for batch_id in np.arange(len(self)):
        self.batch_to_file[batch_id] = self.filenames[self.files_per_batch *
                                                      batch_id:self
                                                      .files_per_batch *
                                                      (batch_id + 1)]
        self.batch_to_samples[batch_id] = sample_indices[:]

  def get_added_channels(self, n_sample: int,
                         da_inputs: xr.DataArray) -> Optional[np.ndarray]:
    """Returns all requested channels to be added to the generator inputs.

    Args:
      n_sample: The number of samples to be generated.
      da_inputs: DataArray containing a batch of n_sample samples of all inputs
        loaded from file.

    Returns:
      The added channels, if any are requested.
    """
    added_channels = []
    if isinstance(self.static_pred, np.ndarray):
      if n_sample != self._batch_size:
        added_channels.append(
            np.repeat(self.static_pred[0, np.newaxis], n_sample, axis=0))
      else:
        added_channels.append(self.static_pred)

    added_channels.extend(
        BaseDataGenerator.get_time_dep_predictors(self,
                                                  da_inputs.sample.values))

    if added_channels:
      return np.concatenate(added_channels, axis=1)

  def generate(
      self, samples: abc.Sequence,
      filename: Union[str,
                      List[str]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generates input and output samples from the data.

    Args:
      samples: Sequence of sample indices to be generated, () or [] to
        include all samples.
      filename: Name of file containing samples to use, or list of files.

    Returns:
      A batch of predictors and targets, indexed in the generator's dataset by
        indices samples.
    """
    if (isinstance(samples, np.ndarray) and samples.size == 0) or (
        (isinstance(samples, list) or isinstance(samples, tuple)) and
        not samples):
      generate_full_ds = True
      filename = self.filenames
    else:
      generate_full_ds = False

    if isinstance(filename, str):
      ds_full = file_util.load_dataset(filename, mode=self.mode)
    elif isinstance(filename, list) or generate_full_ds:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=self._batch_size) as executor:
        f = executor.map(
            functools.partial(file_util.load_dataset, mode=self.mode), filename)
      ds_full = xr.concat(f, dim='sample')
    else:
      raise TypeError('filename must be str or list, not %s' % type(filename))

    if generate_full_ds:
      da_in = ds_full.predictors.sel(**self._input_sel)
      da_out = ds_full.targets.sel(**self._output_sel)
      n_sample = ds_full.sample.size
      samples = np.arange(n_sample)
    else:
      ds = ds_full.isel(sample=samples)
      da_in = ds.predictors.sel(**self._input_sel)
      da_out = ds.targets.sel(**self._output_sel)
      n_sample = len(samples)

    da_in.load()
    da_out.load()
    p = da_in.values
    t = da_out.values

    # Construct predictors
    added_channels = self.get_added_channels(n_sample, da_in)
    if added_channels is not None:
      p = np.concatenate([p, added_channels], axis=1)

    ds_full.close()

    # Remove samples with NaN.
    if self._remove_nan:
      p, t = DLWP.util.delete_nan_samples(p, t)

    # Format spatial shape for convolutions.
    if self.channels_last:
      p = p.transpose(self._transpose)
      t = t.transpose(self._transpose)

    return [p], [t]

  def to_tf_dataset(
      self,
      batch_size: Optional[int] = None,
      input_names: Optional[List[str]] = None,
      output_names: Optional[List[str]] = None) -> tf.data.Dataset:
    """Obtains a tf.Dataset from a ShardedDataGenerator.

    Args:
      batch_size: If int, use a fixed batch size. Will cause an
        error if the last batch of training data does not have the same number
        of samples.
      input_names: List of names for the inputs, to match the model Input
        layer.
      output_names: List of names for the outputs, to match the model output
        layer.

    Returns:
      A tf.data.Dataset.
    """
    # Determine structure of output data
    p, t = self.generate([0], self.filenames[0])

    return BaseDataGenerator.to_tf_dataset(self, p, t, batch_size, input_names,
                                           output_names)


class FullDataGenerator(BaseDataGenerator):
  """Generator of training data from a Dataset stored in a single file.

  The FullDataGenerator provides an efficient option for training models using
  lightweight datasets that can fit into memory, avoiding IO operations during
  the generation process. For more advanced applications where the size of the
  training data is large, consider the ShardedDataGenerator.
  """

  def __init__(self, ds: xr.Dataset, delay_load: bool = False, **kwargs):
    """Initializes a FullDataGenerator.

    Args:
      ds: Dataset to use for batch generation.
      delay_load: if True, delays the loading of the data until the call to
        generate().
      **kwargs: BaseDataGenerator keyword arguments.
    """
    super(FullDataGenerator, self).__init__(**kwargs)
    self.n_sample = ds.dims['sample']
    self.ds = ds
    BaseDataGenerator.set_input_output_sel(self, self.ds)

    self.input_da = self.ds.predictors.sel(**self._input_sel)
    self.output_da = self.ds.targets.sel(**self._output_sel)
    if not delay_load:
      self.input_da.load()
      self.output_da.load()
    self.input_shape = self.input_da.shape[1:]
    self.output_shape = self.output_da.shape[1:]

    self.on_epoch_end()
    BaseDataGenerator.load_static_predictors(self, self.ds)
    if self.verbose:
      logging.info('Generator initialized.')

  def load_time_dep_predictors(self):
    """Loads all requested time-dependent predictors to memory."""
    time_dep_pred = BaseDataGenerator.get_time_dep_predictors(
        self, self.ds.sample.values)

    if self._add_insolation:
      self.insolation_da = time_dep_pred[0]
    if self._add_time:
      self.time_da = time_dep_pred[int(self._add_insolation)]

  def get_added_channels(self, samples: abc.Sequence) -> Optional[np.ndarray]:
    """Returns all requested channels to be added to the generator inputs.

    Args:
      samples: Sequence of sample indices to be generated.

    Returns:
      The added channels, if any are requested.
    """
    n_sample = len(samples)
    added_channels = []
    if isinstance(self.static_pred, np.ndarray):
      if n_sample != self._batch_size:
        added_channels.append(
            np.repeat(self.static_pred[0, np.newaxis], n_sample, axis=0))
      else:
        added_channels.append(self.static_pred)

    if self._add_insolation:
      added_channels.append(self.insolation_da[samples])

    if self._add_time:
      added_channels.append(self.time_da[samples])

    if added_channels:
      return np.concatenate(added_channels, axis=1)

  def on_epoch_end(self):
    """Shuffles samples after each epoch if shuffling is requested."""
    self._indices = np.arange(self.n_sample)
    if self._shuffle:
      np.random.shuffle(self._indices)

  def __getitem__(self,
                  index: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Get one batch of data.

    Args:
      index: Index of requested batch.

    Returns:
      predictors: predictors array.
      targets: targets array.
    Raises:
      IndexError: if index > len(self).
    """
    # Generate indexes of the batch
    if index < 0:
      index = len(self) + index
    if index > len(self):
      raise IndexError
    indices = self._indices[index * self._batch_size:(index + 1) *
                            self._batch_size]
    return self.generate(indices)

  def generate(
      self, samples: abc.Sequence) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generates input and output samples from the data.

    Args:
      samples: Sequence of sample indices to be generated, () or [] to include
        all samples.

    Returns:
      A batch of predictors and targets, indexed in the generator's dataset by
        indices samples.
    """
    if (isinstance(samples, np.ndarray) and samples.size == 0) or (
        (isinstance(samples, list) or isinstance(samples, tuple)) and
        not samples):
      samples = np.arange(self.n_sample, dtype=np.int)

    p = self.input_da.isel(sample=samples).values
    added_channels = self.get_added_channels(samples)
    if added_channels is not None:
      p = np.concatenate([p, added_channels], axis=1)
    t = self.output_da.isel(sample=samples).values

    # Remove samples with NaN; scale and impute
    if self._remove_nan:
      p, t = DLWP.util.delete_nan_samples(p, t)
    # Transpose to channels_last if requested
    if self.channels_last:
      p = p.transpose(self._transpose)
      t = t.transpose(self._transpose)

    return [p], [t]

  def to_tf_dataset(
      self,
      batch_size: Optional[int] = None,
      input_names: Optional[List[str]] = None,
      output_names: Optional[List[str]] = None) -> tf.data.Dataset:
    """Obtains a tf.Dataset from a FullDataGenerator.

    Args:
      batch_size: If int, use a fixed batch size. Will cause an error if the
        last batch of training data does not have the same number of samples.
      input_names: List of names for the inputs, to match the model Input layer.
      output_names: List of names for the outputs, to match the model output
        layer.

    Returns:
      A tf.data.Dataset.
    """
    # Determine structure of output data
    p, t = self.generate([0])

    return BaseDataGenerator.to_tf_dataset(self, p, t, batch_size, input_names,
                                           output_names)
