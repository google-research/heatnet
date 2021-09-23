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
"""Custom loss functions for HeatNet models."""
from typing import Callable

import tensorflow as tf


def channelwise_loss(loss: Callable[[tf.Tensor, tf.Tensor],
                                    tf.Tensor], channel: int,
                     name: str) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
  """Returns a channel-specific version of the given loss.

  Args:
    loss: Generic keras loss function taking two input tf.Tensors.
    channel: Channel to which loss is applied.
    name: Name of the returned loss function.

  Returns:
    The channel-wise loss.
  """

  def channel_loss(y_true, y_pred):
    return loss(y_true[..., channel], y_pred[..., channel])

  channel_loss.__name__ = name
  return channel_loss


def parameterized_loss(
    loss: Callable[[tf.Tensor, tf.Tensor, float], tf.Tensor], parameter: float,
    name: str) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
  """Returns the given parameterized loss evaluated at a parameter value.

  Args:
    loss: Generic loss function taking two input tf.Tensors and a parameter.
    parameter: Numeric value passed to a parameterized loss function.
    name: Name of the returned loss function.

  Returns:
    The loss.
  """

  def param_loss(y_true, y_pred):
    return loss(y_true, y_pred, parameter)

  param_loss.__name__ = name
  return param_loss


def masked_mse(y_true: tf.Tensor,
               y_pred: tf.Tensor,
               mask_below: float = 1.0) -> tf.Tensor:
  """Custom loss applying the MSE to values over a particular threshold.

  The mask is applied independently to true and predicted outputs to
  penalize both false positives and false negatives in values higher
  than mask_below.

  Args:
    y_true: True target.
    y_pred: Predicted value of target or forecast.
    mask_below: Masks all values below mask_below to zero.

  Returns:
    The value of the loss.
  """
  y_true_masked = tf.where(y_true > mask_below, y_true, 0)
  y_pred_masked = tf.where(y_pred > mask_below, y_pred, 0)
  mse = tf.keras.losses.MeanSquaredError()
  return mse(y_true_masked, y_pred_masked)


def mse_exp(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  """Mean squared error of the exponential of the inputs.

  Args:
    y_true: True target.
    y_pred: Predicted value of target or forecast.

  Returns:
    The value of the loss.
  """
  y_true_exp = tf.keras.activations.exponential(y_true)
  y_pred_exp = tf.keras.activations.exponential(y_pred)
  mse = tf.keras.losses.MeanSquaredError()
  return mse(y_true_exp, y_pred_exp)


def mse_exp_negexp(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  """Mean squared error of the exponential of the inputs and minus inputs.

  Args:
    y_true: True target.
    y_pred: Predicted value of target or forecast.

  Returns:
    The value of the loss.
  """
  return 0.5 * (mse_exp(y_true, y_pred) + mse_exp(-y_true, -y_pred))


def mse_negexp(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  """Mean squared error of the exponential of the negative of the inputs.

  Args:
    y_true: True target.
    y_pred: Predicted value of target or forecast.

  Returns:
    The value of the loss.
  """
  return mse_exp(-y_true, -y_pred)


def kl_div_softmaxmin(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  """Custom loss combining spatial softmax/min + channel-wise KL divergence.

  Custom loss that applies a spatial softmax and softmin to data on the cubed
  sphere, and then the Kullback-Liebler divergence. Follows Qi and Majda
  (PNAS, 2019, https://www.pnas.org/content/117/1/52.short). The loss
  assumes a channels_last keras configuration, such that axes (-4, -3, -2)
  correspond to the coordinates (face, height, width) on the cubed sphere.

  Args:
    y_true: True target.
    y_pred: Predicted value of target or forecast.

  Returns:
    The value of the loss.
  """
  y_true_sm_p = tf.keras.activations.softmax(y_true, axis=(-4, -3, -2))
  y_pred_sm_p = tf.keras.activations.softmax(y_pred, axis=(-4, -3, -2))
  y_true_sm_m = tf.keras.activations.softmax(-y_true, axis=(-4, -3, -2))
  y_pred_sm_m = tf.keras.activations.softmax(-y_pred, axis=(-4, -3, -2))
  kl = tf.keras.losses.KLDivergence()
  return 0.5 * (kl(y_true_sm_p, y_pred_sm_p) + kl(y_true_sm_m, y_pred_sm_m))


def sym_kl_div_softmax(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  """Custom loss combining spatial softmax + symmetric KL divergence.

  Custom loss that applies a spatial softmax to data on the cubed sphere,
  and then the symmetrized Kullback-Liebler divergence. The loss
  assumes a channels_last keras configuration.

  Args:
    y_true: True target.
    y_pred: Predicted value of target or forecast.

  Returns:
    The value of the loss.
  """
  y_true_sm = tf.keras.activations.softmax(y_true, axis=(-4, -3, -2))
  y_pred_sm = tf.keras.activations.softmax(y_pred, axis=(-4, -3, -2))
  kl = tf.keras.losses.KLDivergence()
  return 0.5 * (kl(y_true_sm, y_pred_sm) + kl(y_pred_sm, y_true_sm))
