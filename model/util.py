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
"""Utils for keras HeatNet models."""
import json
import os
import time
from typing import Any, Dict, Optional

from absl import logging
import DLWP.util
import heatnet.file_util as file_util
import tensorflow as tf
import xarray as xr


def load_keras_checkpoint(
    ckpt_file_name: str,
    custom_objects: Optional[Dict[str, Any]] = None) -> tf.keras.Model:
  """Loads a model with weights saved with the `ModelCheckpoint()` method.

  The method automatically loads all custom methods and classes in DLWP.custom,
  including convolutional layers for data on the cubed sphere. All generic
  custom losses in heatnet.model.losses are also loaded automatically.

  Args:
    ckpt_file_name: Path to checkpoint model file.
    custom_objects: Any custom functions or classes to be included when Keras
      loads the model. There is no need to add objects in DLWP.custom as those
      are added automatically.

  Returns:
    model: Loaded model object with checkpointed weights.
  """
  # Load the saved keras model weights
  custom_objects = custom_objects or {}
  custom_objects.update(DLWP.util.get_classes('DLWP.custom'))
  custom_objects.update(DLWP.util.get_methods('DLWP.custom'))
  custom_objects.update(DLWP.util.get_classes('heatnet.model.losses'))
  custom_objects.update(DLWP.util.get_methods('heatnet.model.losses'))
  with file_util.mktemp(suffix='.keras') as m:
    tf.io.gfile.copy('%s.keras' % ckpt_file_name, m, overwrite=True)
    return tf.keras.models.load_model(
        m, custom_objects=custom_objects, compile=True)


def checkpoint_keras_model(model: tf.keras.Model,
                           filename: str,
                           include_optimizer: bool = False):
  """Save the Keras model to the given filename in keras format.

  Args:
    model: The keras model to save.
    filename: The name of the file to save to, including the full path, but not
      the extension.
    include_optimizer: Whether the optimizer should also be saved.
  """
  file_util.maybe_make_dirs(filename)
  with file_util.mktemp(suffix='.keras') as f:
    tf.keras.models.save_model(model, f, include_optimizer=include_optimizer)
    tf.io.gfile.copy(f, f'{filename}.keras', overwrite=True)


def save_model(model: tf.keras.Model,
               dirname: str,
               include_optimizer: bool = False):
  """Save the Keras model to the given filename in SavedModel format.

  Args:
    model: The keras model to save.
    dirname: The name of the SavedModel directory to save to, including the full
      path.
    include_optimizer: Whether the optimizer should also be saved.
  """
  with file_util.mkdtemp() as tmp_dir:
    tf.keras.models.save_model(
        model, tmp_dir, include_optimizer=include_optimizer)
    file_util.copy_dir(tmp_dir, dirname, overwrite=True)


class ModelExporter(tf.keras.callbacks.Callback):
  """Custom keras callback to checkpoint HeatNet models."""

  def __init__(self,
               metric_key: str,
               mode: str,
               output_dir: str,
               ckpt_name: str = 'model',
               keras_format: bool = False):
    """Initializes a ModelExporter.

    Args:
      metric_key: Name of the metric to track with the ModelExporter.
      mode: If 'min' or 'max', exports only new best models with respect to the
        condition and the chosen metric. If 'all', checkpoints after every
        epoch.
      output_dir: Path of directory where checkpoints are written to disk.
      ckpt_name: Base name of the checkpoint files to be saved.
      keras_format: If True, the model is stored in .keras format. Otherwise,
        the model is stored in SavedModel format.
    """
    self.metric_key = metric_key
    self.output_dir = output_dir
    assert mode in ('min', 'max', 'all')
    self.mode = mode
    self.best = None
    self.ckpt_name = ckpt_name
    self.keras_format = keras_format

  def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
    """Loads best metric, if found in the output_dir."""
    if self.best is not None or self.mode == 'all':
      return
    # Check whether to restore best from saved metrics file.
    metrics_file_pattern = os.path.join(self.output_dir, 'metrics_*.json')
    metrics_file_list = sorted(tf.io.gfile.glob(metrics_file_pattern))
    if metrics_file_list:
      latest_metrics_filepath = metrics_file_list[-1]
      with tf.io.gfile.GFile(latest_metrics_filepath, 'r') as f:
        metrics_dict = json.load(f)
        best_val_txt = metrics_dict[self.metric_key]
        if best_val_txt:
          self.best = float(best_val_txt)
          logging.info('Best metric value (%s=%s) restored on_train_begin.',
                       self.metric_key, self.best)

  def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
    """Saves model and metrics to file, if the mode condition is met."""
    metric = logs[self.metric_key]
    if (self.best is None or (self.mode == 'min' and metric < self.best) or
        (self.mode == 'max' and metric > self.best)):
      timestamp = str(int(time.time()))
      filename = f'{self.ckpt_name}_{str(epoch)}_{timestamp}'
      if self.keras_format:
        checkpoint_keras_model(
            self.model,
            os.path.join(self.output_dir, filename),
            include_optimizer=True)
      else:
        save_model(
            self.model,
            os.path.join(self.output_dir, filename),
            include_optimizer=True)
      if self.mode != 'all':
        logging.info(
            'Model weights saved as %s after epoch %d (%s=%s, '
            'previous best=%s)', filename, epoch, self.metric_key, metric,
            self.best)
        self.best = metric
      else:
        logging.info('Model saved after epoch %d (%s=%s)', epoch,
                     self.metric_key, metric)
      # Save metrics and epoch.
      metrics_filename = os.path.join(self.output_dir,
                                      f'metrics_{filename}.json')
      # Make ndarrays/tf.tensors JSON serializable.
      for metric_key in logs:
        if hasattr(logs[metric_key], 'dtype'):
          if tf.is_tensor(logs[metric_key]):
            logs[metric_key] = logs[metric_key].numpy()
          logs[metric_key] = logs[metric_key].item()
      metrics_dict = logs
      metrics_dict['epoch'] = epoch
      file_util.save_dict_to_json(metrics_dict, metrics_filename)


def rename_forecast_dims(forecast_ds: xr.Dataset) -> xr.Dataset:
  """Rename spatial forecast dimensions.

  Args:
    forecast_ds: Forecast dataset with spatial dimensions on the cubed sphere.

  Returns:
    A dataset with renamed dimensions.
  """
  return forecast_ds.rename_dims({
      'x0': 'face',
      'x1': 'height',
      'x2': 'width'
  }).rename_vars({
      'x0': 'face',
      'x1': 'height',
      'x2': 'width'
  })
