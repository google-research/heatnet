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
"""HeatNet model architectures."""
from typing import Tuple, List

from DLWP.custom import CubeSphereConv2D
from DLWP.custom import CubeSpherePadding2D
import tensorflow as tf


class HeatnetArchitecture(object):
  """Base class for heatnet convolutional architectures."""

  def __init__(self,
               input_sample_shape: Tuple[int, ...],
               output_sample_shape: Tuple[int, ...],
               base_filter_number: Tuple[int, ...],
               layers_per_level: Tuple[int, ...] = (4, 4, 4),
               kernel_size: int = 3,
               filter_factors: Tuple[int, ...] = (1, 2, 4),
               dil_base: int = 2,
               data_format: str = 'channels_last',
               l1_reg: float = 0.0,
               l2_reg: float = 0.0,
               dropouts: Tuple[float, ...] = (0., 0., 0.),
               pooling: str = 'max',
               indep_poles: bool = False,
               batch_norm: bool = False,
               kernel_initializer: str = 'glorot_uniform'):
    """Initializes a Heatnet architecture.

    Args:
      input_sample_shape: Shape of the input samples.
      output_sample_shape: Shape of the output samples.
      base_filter_number: Number of base channels/filter used in the main U-Net
        level.
      layers_per_level: Number of layers in every level of the architecture.
      kernel_size: Size of the convolutional kernels.
      filter_factors: Growth factor of filter number from the preceding to the
        current level.
      dil_base: Base for the power-law growth of dilations with depth.
      data_format: Channel position for keras configuration.
      l1_reg: Magnitude of l1 regularization hyperparameter.
      l2_reg: Magnitude of l2 regularization hyperparameter.
      dropouts: Dropout probability after activations in each level.
      pooling: Type of pooling to apply when downsampling, 'avg' or 'max'.
      indep_poles: If True, weights are independent in the north and south
        poles. Else, they are the same weights and the north pole is flipped.
      batch_norm: If True, applies BatchNorm layers after each nonlinearity.
      kernel_initializer: The initializer used for the convolutional weights.
    """
    if batch_norm and any(dropouts):
      raise TypeError('Dropout and BatchNormalization layers should not be'
                      'implemented at the same time.')
    if pooling not in ['avg', 'max']:
      raise ValueError(f'pooling should be arg or max, not {pooling}.')

    self.input_sample_shape = input_sample_shape
    self.output_sample_shape = output_sample_shape
    self.base_filter_number = base_filter_number
    self.dil_base = dil_base
    self.kernel_size = kernel_size
    self.pooling = pooling
    self.layers = layers_per_level
    # Encoder layers
    self.encoder_layers = (self.layers[0] // 2, self.layers[1] // 2,
                           self.layers[2])
    self.filter_factors = filter_factors
    self.dropouts = dropouts
    self.batch_norm = batch_norm
    # Keras conv kwargs
    self.conv_kwargs = {
        'padding': 'valid',
        'activation': 'linear',
        'data_format': data_format,
        'kernel_regularizer': tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
        'flip_north_pole': not indep_poles,
        'independent_north_pole': indep_poles,
        'kernel_initializer': kernel_initializer,
    }
    self.data_format = data_format
    self.main_input = tf.keras.layers.Input(
        shape=input_sample_shape, name='main_input')

  def prelu(self) -> tf.keras.layers.PReLU:
    """Parametric ReLU activation that shares weights across space."""
    if self.data_format == 'channels_last':
      return tf.keras.layers.PReLU(shared_axes=[1, 2, 3])
    elif self.data_format == 'channels_first':
      return tf.keras.layers.PReLU(shared_axes=[2, 3, 4])
    else:
      raise ValueError(f'Data format {self.data_format} not recognized.')

  def cube_padding(self, dilation: int) -> CubeSpherePadding2D:
    """Padding layer for 2D dilated convolutions on the cubed sphere.

    Args:
      dilation: Dilation rate used in the upcoming convolution.

    Returns:
      The padding layer.
    """
    return CubeSpherePadding2D(
        int(dilation * (self.kernel_size - 1) / 2),
        data_format=self.data_format)

  def pool_layer(self, pool_size: int) -> tf.keras.layers.AveragePooling3D:
    """Pooling layer on the cubed sphere.

    Args:
      pool_size: Pooling rate.

    Returns:
      The pooling layer.
    """
    if self.pooling == 'avg':
      return tf.keras.layers.AveragePooling3D((1, pool_size, pool_size),
                                              data_format=self.data_format)
    elif self.pooling == 'max':
      return tf.keras.layers.MaxPooling3D((1, pool_size, pool_size),
                                          data_format=self.data_format)  # pytype: disable=bad-return-type  # typed-keras

  def upsample_layer(self, ups_size: int) -> tf.keras.layers.UpSampling3D:
    """Upsampling layer on the cubed sphere.

    Args:
      ups_size: Upsampling rate.

    Returns:
      The upsampling layer.
    """
    return tf.keras.layers.UpSampling3D((1, ups_size, ups_size),
                                        data_format=self.data_format)

  def dilated_conv_level(self, level: int, dilation: int) -> CubeSphereConv2D:
    """2D dilated convolution on the cubed sphere.

    Args:
      level: Encoder level where the layer is applied.
      dilation: The dilation rate applied to the convolution.

    Returns:
      The specified CubeSphereConv2D layer.
    """
    return CubeSphereConv2D(
        self.filter_factors[level] * self.base_filter_number,
        self.kernel_size,
        dilation_rate=dilation,
        **self.conv_kwargs)

  def batch_norm_dropout(self, level: int, x: tf.Tensor) -> tf.Tensor:
    """Adds a level-specific BatchNorm or Dropout layer.

    Args:
      level: Encoder level where the layer is applied.
      x: Input tensor to the layer.

    Returns:
      Output of the layer.
    """
    if self.batch_norm:
      return tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    elif self.dropouts[level]:
      return tf.keras.layers.Dropout(self.dropouts[level])(x)
    else:
      return x

  def encoder_stack(self, level: int, x: tf.Tensor) -> tf.Tensor:
    """Adds a level-specific encoder stack.

    Args:
      level: Encoder level.
      x: Input tensor to the encoder stack.

    Returns:
      Output of the stack.
    """
    for layer_i in range(self.encoder_layers[level]):
      x = self.cube_padding(self.dil_base**layer_i)(x)
      x = self.prelu()(
          self.dilated_conv_level(level, self.dil_base**layer_i)(x))
      x = self.batch_norm_dropout(level, x)
    return x

  def decoder_stack(self,
                    level: int,
                    x: tf.Tensor,
                    dil_off: int = 0) -> tf.Tensor:
    """Adds a level-specific decoder stack.

    Args:
      level: Encoder level.
      x: Input tensor to the encoder stack.
      dil_off: An offset in the dilation power with respect to the index of
        layers in the current level.

    Returns:
      Output of the stack.
    """
    for layer_i in range(self.encoder_layers[level], self.layers[level]):
      x = self.cube_padding(self.dil_base**(layer_i + dil_off))(x)
      x = self.prelu()(
          self.dilated_conv_level(level, self.dil_base**(layer_i + dil_off))(x))
      x = self.batch_norm_dropout(level, x)
    return x

  def conv_final(self):
    return CubeSphereConv2D(
        self.output_sample_shape[-1], 1, name='output', **self.conv_kwargs)


class Heatnet3Plus(HeatnetArchitecture):
  """Heatnet architecture based on the UNet3+ of Huang et al (2020)."""

  def forward_model(self, x: tf.Tensor) -> List[tf.Tensor]:
    """Forward model of Heatnet3Plus."""
    x0 = x
    # Persistence skip
    xres0 = self.cube_padding(1)(x0)
    xres0 = self.prelu()(self.dilated_conv_level(0, 1)(xres0))

    # Top level
    x0 = self.encoder_stack(0, x0)

    # Second level (Encoder)
    x1 = self.pool_layer(2)(x0)
    x1 = self.encoder_stack(1, x1)

    # Third level (Encoder)
    x2 = self.pool_layer(2)(x1)
    x2 = self.encoder_stack(2, x2)

    # Feed all encoder levels to the 2nd level of the decoder
    x2_21 = self.upsample_layer(2)(x2)
    x2_21 = self.cube_padding(1)(x2_21)
    x2_21 = self.prelu()(self.dilated_conv_level(0, 1)(x2_21))
    x1_11 = self.cube_padding(self.dil_base**self.encoder_layers[1])(x1)
    x1_11 = self.prelu()(
        self.dilated_conv_level(0,
                                self.dil_base**self.encoder_layers[1])(x1_11))
    x0_01 = self.pool_layer(2)(x0)
    x0_01 = self.cube_padding(self.dil_base**self.encoder_layers[0])(x0_01)
    x0_01 = self.prelu()(
        self.dilated_conv_level(0,
                                self.dil_base**self.encoder_layers[0])(x0_01))
    # Second level (Decoder)
    x_dec_1 = tf.keras.layers.concatenate([x2_21, x1_11, x0_01], axis=-1)
    x_dec_1 = self.decoder_stack(1, x_dec_1, dil_off=1)

    # Feed 1st and 3rd encoder levels, 2nd decoder level to 1st decoder level.
    x_dec_1_10 = self.upsample_layer(2)(x_dec_1)
    x_dec_1_10 = self.cube_padding(1)(x_dec_1_10)
    x_dec_1_10 = self.prelu()(self.dilated_conv_level(0, 1)(x_dec_1_10))
    x2_20 = self.upsample_layer(4)(x2)
    x2_20 = self.cube_padding(1)(x2_20)
    x2_20 = self.prelu()(self.dilated_conv_level(0, 1)(x2_20))
    x0_00 = self.cube_padding(self.dil_base**self.encoder_layers[0])(x0)
    x0_00 = self.prelu()(
        self.dilated_conv_level(0,
                                self.dil_base**self.encoder_layers[0])(x0_00))
    # First level (Decoder)
    x_dec_0 = tf.keras.layers.concatenate([x_dec_1_10, x0_00, x2_20], axis=-1)
    x_dec_0 = self.decoder_stack(0, x_dec_0, dil_off=1)
    # Concatenate persistence skip connection
    x = tf.keras.layers.concatenate([x_dec_0, xres0], axis=-1)
    x = self.conv_final()(x)
    return [x]

  def get_model(self) -> tf.keras.Model:
    return tf.keras.Model(
        inputs=self.main_input, outputs=self.forward_model(self.main_input))


class HeatUnet(HeatnetArchitecture):
  """Heatnet architecture based on the basic UNet architecture."""

  def __init__(self, *args, **kwargs):
    """Initializes a HeatUnet architecture.

    Args:
      *args: HeatnetArchitecture arguments.
      **kwargs: HeatnetArchitecture keyword arguments.
    """
    super(HeatUnet, self).__init__(*args, **kwargs)

    # Layers feeding upper decoders are defined differently, so remove.
    self.encoder_layers = (self.layers[0] // 2, self.layers[1] // 2,
                           self.layers[2] - 1)
    self.layers = (self.layers[0], self.layers[1] - 1, self.layers[2] - 1)

  def decoder_feed(self, level: int, x: tf.Tensor) -> tf.Tensor:
    """Adds a level-specific decoder feed layer.

    This layer halves the number of channels with respect to the rest
    of the layers in the level, and upsamples the result.

    Args:
      level: Level on the input tensor.
      x: Input tensor to the decoder feed layer.

    Returns:
      Output of the decoder feed layer.
    """
    x = self.cube_padding(self.dil_base**(self.layers[level]))(x)
    x = self.prelu()(
        self.dilated_conv_level(level - 1,
                                self.dil_base**(self.layers[level]))(x))
    x = self.batch_norm_dropout(level, x)
    return self.upsample_layer(2)(x)

  def forward_model(self, x: tf.Tensor) -> List[tf.Tensor]:
    """Forward model of HeatUnet."""
    x0 = x
    # Persistence skip
    xres0 = self.cube_padding(1)(x0)
    xres0 = self.prelu()(self.dilated_conv_level(0, 1)(xres0))
    # Top level
    x0 = self.encoder_stack(0, x0)

    # Second level (Encoder)
    x1 = self.pool_layer(2)(x0)
    x1 = self.encoder_stack(1, x1)

    # Third level (Encoder)
    x2 = self.pool_layer(2)(x1)
    x2 = self.encoder_stack(2, x2)
    x2 = self.decoder_feed(2, x2)

    # Feed 2nd and 3rd encoder levels to 2nd decoder level.
    x_dec_1 = tf.keras.layers.concatenate([x2, x1], axis=-1)
    x_dec_1 = self.decoder_stack(1, x_dec_1)
    x_dec_1 = self.decoder_feed(1, x_dec_1)

    # Feed 2nd decoder and 1st encoder levels to 1st decoder level.
    x_dec_0 = tf.keras.layers.concatenate([x_dec_1, x0], axis=-1)
    x_dec_0 = self.decoder_stack(0, x_dec_0)

    x = tf.keras.layers.concatenate([x_dec_0, xres0], axis=-1)
    x = self.conv_final()(x)
    return [x]

  def get_model(self) -> tf.keras.Model:
    return tf.keras.Model(
        inputs=self.main_input, outputs=self.forward_model(self.main_input))
