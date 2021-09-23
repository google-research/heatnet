"""Tests for functions in model/util.py."""
import os

from absl.testing import absltest
import heatnet.file_util as file_util
import heatnet.model
import heatnet.model.util as model_util
import numpy as np
import tensorflow as tf


class ModelUtilTest(absltest.TestCase):
  """Tests for model utils."""

  def test_save_load_model(self):
    """Tests save_model, checkpointing and loading."""
    inp_shape = (6, 48, 48, 2)
    out_shape = (6, 48, 48, 1)
    batch = 2
    x = tf.random.normal((batch,) + inp_shape)
    model = heatnet.model.Heatnet3Plus(inp_shape, out_shape, 2).get_model()
    model.compile()

    with file_util.mkdtemp() as tmp_dir:
      filename = os.path.join(tmp_dir, 'tmp_model')
      model_util.save_model(model, filename)
      # Checks that SavedModel directory has been saved.
      self.assertLen(tf.io.gfile.listdir(tmp_dir), 1)

      model_util.checkpoint_keras_model(model, filename)
      # Checks that keras model has been saved.
      self.assertLen(tf.io.gfile.listdir(tmp_dir), 2)

      loaded_model = tf.keras.models.load_model(filename)
      loaded_ckpt = model_util.load_keras_checkpoint(filename)
      # Checks that saved and loaded model have the same forward model.
      self.assertTrue(np.allclose(model.predict(x), loaded_model.predict(x)))
      self.assertTrue(np.allclose(model.predict(x), loaded_ckpt.predict(x)))


class ModelExporterTest(absltest.TestCase):
  """Tests the ModelExporter."""

  def test_model_exporter(self):
    """Tests correct checkpointing."""
    inp_shape = (6, 48, 48, 2)
    out_shape = (6, 48, 48, 1)
    batch = 1
    x = tf.random.normal((batch,) + inp_shape)
    y = tf.random.normal((batch,) + out_shape)
    model = heatnet.model.Heatnet3Plus(inp_shape, out_shape, 2).get_model()
    model.compile(loss='mse', metrics=['mse'])

    with file_util.mkdtemp() as tmp_dir:
      checkpoint_1 = model_util.ModelExporter(
          'loss', 'min', tmp_dir, keras_format=True)
      checkpoint_2 = model_util.ModelExporter(
          'loss', 'max', tmp_dir, ckpt_name='ckpt_max', keras_format=True)
      model.fit(x, y, epochs=2, callbacks=[checkpoint_1, checkpoint_2])
      # checkpoint_2 should not write files after the second epoch.
      self.assertLen(tf.io.gfile.listdir(tmp_dir), 6)


if __name__ == '__main__':
  absltest.main()
