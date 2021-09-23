"""Tests for functions and classes in model/models.py."""
from absl.testing import absltest
import heatnet.model
import tensorflow as tf


class HeatnetArchitectureTest(absltest.TestCase):
  """Tests for HeatnetArchitecture."""

  def test_init(self):
    """Tests correct initialization."""
    arch = heatnet.model.HeatnetArchitecture((), (), 1)
    self.assertEmpty(arch.input_sample_shape)
    self.assertEmpty(arch.output_sample_shape)
    for level, dim in enumerate(arch.layers):
      self.assertGreaterEqual(dim, arch.encoder_layers[level])


class Heatnet3PlusTest(absltest.TestCase):
  """Tests for Heatnet3Plus."""

  def test_model(self):
    """Tests correct model behavior."""
    inp_shape = (6, 48, 48, 2)
    out_shape = (6, 48, 48, 3)
    batch = 8
    arch = heatnet.model.Heatnet3Plus(inp_shape, out_shape, 4)
    model = arch.get_model()
    model.compile()
    y = model.predict(tf.random.normal((batch,) + inp_shape))
    self.assertEqual(all(tf.shape(y)), all((batch,) + out_shape))


class HeatUnetTest(absltest.TestCase):
  """Tests for HeatUnet."""

  def test_model(self):
    """Tests correct model behavior."""
    inp_shape = (6, 48, 48, 2)
    out_shape = (6, 48, 48, 3)
    batch = 8
    arch = heatnet.model.HeatUnet(inp_shape, out_shape, 4)
    model = arch.get_model()
    model.compile()
    y = model.predict(tf.random.normal((batch,) + inp_shape))
    self.assertEqual(all(tf.shape(y)), all((batch,) + out_shape))


if __name__ == '__main__':
  absltest.main()
