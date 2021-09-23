"""Tests for functions and classes in model/losses.py."""
from absl.testing import absltest
from heatnet.model.losses import channelwise_loss
from heatnet.model.losses import kl_div_softmaxmin
from heatnet.model.losses import masked_mse
from heatnet.model.losses import mse_exp
from heatnet.model.losses import mse_exp_negexp
from heatnet.model.losses import mse_negexp
from heatnet.model.losses import parameterized_loss
from heatnet.model.losses import sym_kl_div_softmax
import tensorflow as tf


class ChannelwiseLossTest(absltest.TestCase):
  """Tests for channelwise_loss functions."""

  def test_channels(self):
    """Tests correct channel selection."""
    x = tf.concat([tf.constant(5, shape=(2, 2)),
                   tf.constant(1, shape=(2, 2))],
                  axis=-1)
    y = tf.concat([tf.constant(3, shape=(2, 2)),
                   tf.constant(8, shape=(2, 2))],
                  axis=-1)
    mse_0 = channelwise_loss(tf.keras.losses.MeanSquaredError(), 0, 'mse_0')
    mse_1 = channelwise_loss(tf.keras.losses.MeanSquaredError(), -1, 'mse_1')
    self.assertEqual(mse_0(x, y).numpy(), 4)
    self.assertEqual(mse_1(x, y).numpy(), 49)


class ParameterizedLossTest(absltest.TestCase):
  """Tests for parameterized_loss functions."""

  def test_parameter(self):
    """Tests consistent parameter behavior."""
    x = tf.random.normal((2, 6, 24, 24, 2))
    x = tf.where(x < 0, 10 * x, x)
    y = tf.random.normal((2, 6, 24, 24, 2))
    y = tf.where(y < 0, 10 * y, y)
    mse_0 = parameterized_loss(masked_mse, 0.0, 'mse_0')
    mse_1 = parameterized_loss(masked_mse, -30.0, 'mse_1')
    self.assertGreater(mse_1(x, y), mse_0(x, y))


class CustomLossTest(absltest.TestCase):
  """Tests for custom loss functions."""

  def test_mse_exp_losses(self):
    """Tests exponential-based losses."""
    x = tf.random.normal((2, 6, 24, 24, 2))
    x = tf.where(x < 0, 10 * x, x)
    y = tf.random.normal((2, 6, 24, 24, 2))
    y = tf.where(y < 0, 10 * y, y)
    mse = tf.keras.losses.MeanSquaredError()
    self.assertGreater(mse_exp_negexp(x, y), mse(x, y))
    self.assertGreater(mse_exp_negexp(x, y), mse_exp(x, y))
    self.assertGreater(mse_negexp(x, y), mse_exp(x, y))

  def test_softmax_losses(self):
    """Tests softmax-related losses."""
    x = tf.random.normal((2, 6, 24, 24, 2))
    y = tf.random.normal((2, 6, 24, 24, 2))
    self.assertGreater(kl_div_softmaxmin(x, y), 0)
    self.assertGreater(sym_kl_div_softmax(x, y), 0)


if __name__ == '__main__':
  absltest.main()
