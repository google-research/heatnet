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
"""Estimator classes for heatnet models."""
import collections.abc as abc
import copy
from typing import List, Tuple, Optional, Union

import heatnet.data
import heatnet.file_util as file_util
import heatnet.model.util as hm_util
import numpy as np
import tensorflow as tf
import xarray as xr


class BaseEstimator(object):
  """Class to perform model evaluations with consistent metadata."""

  def __init__(self,
               model: tf.keras.Model,
               generator: heatnet.data.BaseDataGenerator,
               include_targets: bool = False,
               include_predictors: bool = False):
    """Initializes a BaseEstimator from a model and a generator.

    Args:
      model: keras model instance.
      generator: Instance of a subclass of BaseDataGenerator.
      include_targets: If True, the true targets are also included as outputs of
        the estimator.
      include_predictors: If True, the predictors are also included as outputs
        of the estimator.
    """

    self.model = model
    self.generator = generator
    self.include_predictors = include_predictors
    self.include_targets = include_targets
    if self.generator._shuffle:
      raise ValueError('Only Generators with shuffle=False should be'
                       'passed to the Estimator.')
    self.rank = self.generator.rank
    self._output_sel = {
        k: np.array(v) for k, v in self.generator._output_sel.items()
    }
    self._input_sel = {
        k: np.array(v) for k, v in self.generator._input_sel.items()
    }
    # Keep primitive inputs
    self._primitive_inputs = copy.deepcopy(self._input_sel['pred_varlev'])
    # Strict order of added channels to conform with generators.
    if self.generator._add_latlon:
      self._input_sel['pred_varlev'] = np.concatenate([
          self._input_sel['pred_varlev'],
          np.array(['sin_latitude']),
          np.array(['longitude'])
      ])
    if self.generator._topo_file is not None:
      self._input_sel['pred_varlev'] = np.concatenate(
          [self._input_sel['pred_varlev'],
           np.array(['topography'])])
    if self.generator._lsm_file is not None:
      self._input_sel['pred_varlev'] = np.concatenate(
          [self._input_sel['pred_varlev'],
           np.array(['lsm'])])
    if self.generator._add_insolation:
      self._input_sel['pred_varlev'] = np.concatenate(
          [self._input_sel['pred_varlev'],
           np.array(['insol'])])
    if self.generator._add_time:
      self._input_sel['pred_varlev'] = np.concatenate(
          [self._input_sel['pred_varlev'],
           np.array(['time'])])
    # Channels last option
    self.channels_last = hasattr(
        self.generator, 'channels_last') and self.generator.channels_last

  def add_metadata(self,
                   pred_or_tgt: Union[np.ndarray, abc.Sequence],
                   sample_coord: np.ndarray,
                   data_type: str = 'predictors') -> xr.DataArray:
    """Adds metadata to generated input or output data.

    Args:
      pred_or_tgt: Numpy array of input or output data.
      sample_coord: Sample coordinates corresponding to data samples.
      data_type: Type of data to be processed, 'predictors' or 'targets'.

    Returns:
      pt_da: The predictors/targets as a DataArray with metadata.
    Raises:
      ValueError: If data_type is not 'predictors' or 'targets'.
    """
    if data_type == 'predictors':
      channels = 'pred_varlev'
      selection = self._input_sel[channels]
    elif data_type == 'targets':
      channels = 'tgt_varlev'
      selection = self._output_sel[channels]
    else:
      raise ValueError('data_type must be predictors or targets, but'
                       f'{data_type} was given.')

    pt = pred_or_tgt[0] if isinstance(pred_or_tgt,
                                      abc.Sequence) else pred_or_tgt

    if self.channels_last:
      return xr.DataArray(
          pt,
          coords=([
              sample_coord,
          ] + [
              np.arange(d)
              for d in self.generator.convolution_shape[-self.rank - 1:-1]
          ] + [selection]),
          dims=[
              'time',
          ] + ['x%d' % d for d in range(self.rank)] + [channels],
          name=data_type)
    else:
      return xr.DataArray(
          pt,
          coords=([sample_coord, selection] + [
              np.arange(d)
              for d in self.generator.convolution_shape[-self.rank:]
          ]),
          dims=['time', channels] + ['x%d' % d for d in range(self.rank)],
          name=data_type)

  def forecast_with_metadata(self, pred_da: xr.DataArray,
                             sample_coord: np.ndarray,
                             **kwargs) -> xr.DataArray:
    """Performs model evaluations and adds metadata to the result.

    Args:
      pred_da: DataArray of input/predictor data.
      sample_coord: Sample coordinates corresponding to input samples.
      **kwargs: keyword arguments passed to predict() method.

    Returns:
      The model forecast as a DataArray with metadata.
    """
    # Perform model prediction
    result = self.model.predict(pred_da.values, **kwargs)

    # Return a DataArray.
    rv = result.view()
    if self.channels_last:
      rv.shape = ((len(sample_coord),) +
                  self.generator.output_convolution_shape[-self.rank - 1:-1] +
                  (-1,))
      result_da = xr.DataArray(
          rv,
          coords=[
              sample_coord,
          ] + [
              np.arange(d)
              for d in self.generator.output_convolution_shape[-self.rank -
                                                               1:-1]
          ] + [self._output_sel['tgt_varlev']],
          dims=['time'] + ['x%d' % d for d in range(self.rank)] +
          ['tgt_varlev'],
          name='forecast')
    else:
      rv.shape = ((
          len(sample_coord),
          -1,
      ) + self.generator.output_convolution_shape[-self.rank:])
      result_da = xr.DataArray(
          rv,
          coords=[
              sample_coord,
              self._output_sel['tgt_varlev'],
          ] + [
              np.arange(d)
              for d in self.generator.output_convolution_shape[-self.rank:]
          ],
          dims=['time', 'tgt_varlev'] + ['x%d' % d for d in range(self.rank)],
          name='forecast')
    return result_da

  def scale_targets(
      self,
      scale_file: str,
      forecast_da: xr.DataArray,
  ) -> xr.DataArray:
    """Scales back forecasts or targets using stored mean and std.

    Args:
      scale_file: Path to Dataset containing mean and standard deviation of all
        target fields.
      forecast_da: DataArray containing the normalized forecast or target
        variables.

    Returns:
      forecast_da: The forecast fields, multiplied by their stored std and added
        to their stored mean, as a DataArray with metadata.
    """
    with tf.io.gfile.GFile(scale_file, 'rb') as f:
      with xr.open_dataset(f, engine='h5netcdf') as scale_ds:
        sel_mean = scale_ds['tgt_mean']
        sel_std = scale_ds['tgt_std']
        for target in forecast_da.tgt_varlev.values:
          target_var = '/'.join(target.split('/')[:2])
          forecast_da.loc[:, target] = (
              forecast_da.loc[:, target] * sel_std.loc[target_var] +
              sel_mean.loc[target_var])

      return forecast_da

  def scale_predictors(
      self,
      scale_file: str,
      predictor_da: xr.DataArray,
  ) -> xr.DataArray:
    """Scales back predictors using stored mean and std.

    Args:
      scale_file: Path to Dataset containing mean and standard deviation of all
        target fields.
      predictor_da: DataArray containing the normalized input variables.

    Returns:
      predictor_da: The predictors, multiplied by their stored std and added
        to their stored mean, as a DataArray with metadata.
    """
    with tf.io.gfile.GFile(scale_file, 'rb') as f:
      with xr.open_dataset(f, engine='h5netcdf') as scale_ds:
        sel_mean = scale_ds['pred_mean']
        sel_std = scale_ds['pred_std']
        for predictor in self._primitive_inputs:
          predictor_var = '/'.join(predictor.split('/')[:2])
          predictor_da.loc[:, predictor] = (
              predictor_da.loc[:, predictor] * sel_std.loc[predictor_var] +
              sel_mean.loc[predictor_var])

    return predictor_da

  def predict(self,
              predictors: List[np.ndarray],
              targets: Optional[List[np.ndarray]],
              sample_coord: np.ndarray,
              scale_file: Optional[str] = None,
              rename_dims: bool = True,
              **kwargs) -> Union[xr.Dataset, Tuple[xr.Dataset, ...]]:
    """Returns the model prediction for input data in 'samples', with metadata.

    If 'scale_file' is given, scales back the predictions using the provided
    mean and standard deviation of each variable.

    Args:
      predictors: Generated inputs/predictors.
      targets: Targets corresponding to generated inputs/predictors.
      sample_coord: Sample coordinates corresponding to data samples.
      scale_file: Path to netcdf file containing mean and standard deviation of
        the preprocessed targets. If given, the predictions are scaled using
        mean and standard deviation.
      rename_dims: If True, rename anonymous spatial dimensions to 'face',
        'height', 'width'.
      **kwargs: keyword arguments passed to keras.predict().

    Returns:
      results_ds: Model evaluation of shape ('time', 'tgt_varlev', 'x0'/'face',
        'x1'/'height', 'x2'/'width'), together with true targets if included.
      p_ds: Inputs to the model used to generate evaluations in results_ds, if
        included.
    """
    # Add metadata.
    p_da = self.add_metadata(predictors, sample_coord)
    if targets is not None:
      t_da = self.add_metadata(targets, sample_coord, data_type='targets')
      if self.channels_last:
        t_da = t_da.transpose('time', 'tgt_varlev', ...)

    result_da = self.forecast_with_metadata(p_da, sample_coord, **kwargs)
    # Transpose to channel_first
    if self.channels_last:
      result_da = result_da.transpose('time', 'tgt_varlev', ...)
      p_da = p_da.transpose('time', 'pred_varlev', ...)

    # Scale predictions if scale file is provided
    if scale_file is not None:
      result_da = self.scale_targets(scale_file, result_da)
      if targets is not None:
        t_da = self.scale_targets(scale_file, t_da)
      if self.include_predictors:
        p_da = self.scale_predictors(scale_file, p_da)

    if targets is not None:
      result_ds = xr.merge(
          [result_da.astype(dtype=np.float32),
           t_da.astype(dtype=np.float32)],
          join='exact')
    else:
      result_ds = result_da.astype(dtype=np.float32).to_dataset()
    if rename_dims:
      result_ds = hm_util.rename_forecast_dims(result_ds)

    if self.include_predictors:
      p_ds = p_da.astype(dtype=np.float32).to_dataset()
      if rename_dims:
        p_ds = hm_util.rename_forecast_dims(p_ds)
      return result_ds, p_ds
    else:
      return result_ds


class FullDataEstimator(BaseEstimator):
  """Performs model evaluations from a heatnet.data.FullDataGenerator."""

  def __init__(self, model: tf.keras.Model,
               generator: heatnet.data.FullDataGenerator, **kwargs):
    """Initializes a FullDataEstimator from a model and a FullDataGenerator.

    Args:
      model: keras model instance.
      generator: Instance of a FullDataGenerator.
      **kwargs: Keyword arguments of the BaseEstimator.
    """
    super(FullDataEstimator, self).__init__(model, generator, **kwargs)

  def predict(self, samples: abc.Sequence = (),
              **kwargs) -> Union[xr.Dataset, Tuple[xr.Dataset, ...]]:
    """Performs model evaluations for the given sample indices.

    Args:
      samples: Sequence of sample indices to be generated, () or [] to include
        all samples.
      **kwargs: keyword arguments passed to BaseEstimator.predict().

    Returns:
      results_ds: Model evaluation of shape ('time', 'tgt_varlev', 'x0'/'face',
        'x1'/'height', 'x2'/'width'), together with true targets if included.
      p_ds: Inputs to the model used to generate evaluations in results_ds, if
        included.
    """

    # Load data from the generator
    predictors, targets = self.generator.generate(samples)

    if (isinstance(samples, np.ndarray) and
        samples.size == 0) or (isinstance(samples, abc.Sequence) and
                               not samples):
      sample_coord = self.generator.ds.sample[:]
    else:
      sample_coord = self.generator.ds.sample[samples]

    if self.include_targets:
      return BaseEstimator.predict(self, predictors, targets, sample_coord,
                                   **kwargs)
    else:
      return BaseEstimator.predict(self, predictors, None, sample_coord,
                                   **kwargs)


class ShardedDataEstimator(BaseEstimator):
  """Performs model evaluations from a heatnet.data.ShardedDataGenerator."""

  def __init__(self, model: tf.keras.Model,
               generator: heatnet.data.ShardedDataGenerator, **kwargs):
    """Initializes a ShardedDataEstimator from a model and a generator.

    Args:
      model: keras model instance.
      generator: Instance of a ShardedDataGenerator.
      **kwargs: Keyword arguments of the BaseEstimator.
    """
    super(ShardedDataEstimator, self).__init__(model, generator, **kwargs)

  def predict(self, **kwargs) -> Union[xr.Dataset, Tuple[xr.Dataset, ...]]:
    """Performs model evaluations using all generated data.

    Unlike for the FullDataEstimator, the selection of samples to be evaluated
    is performed at the Generator level, since ShardedDataGenerator can contain
    subsets of the full dataset.

    Args:
      **kwargs: keyword arguments passed to BaseEstimator.predict().

    Returns:
      results_ds: Model evaluation of shape ('time', 'tgt_varlev', 'x0'/'face',
        'x1'/'height', 'x2'/'width'), together with true targets if included.
      p_ds: Inputs to the model used to generate evaluations in results_ds, if
        included.
    """
    # Load data from the generator
    predictors, targets = self.generator.generate((), self.generator.filenames)

    sample_coord = xr.concat([
        file_util.load_dataset(file, mode='ext')
        for file in self.generator.filenames
    ],
                             dim='sample').sample

    if self.include_targets:
      return BaseEstimator.predict(self, predictors, targets, sample_coord,
                                   **kwargs)
    else:
      return BaseEstimator.predict(self, predictors, None, sample_coord,
                                   **kwargs)
