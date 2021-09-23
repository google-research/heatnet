"""Tests for functions in file_util.py."""
import os

from absl.testing import absltest
import heatnet.file_util as file_util
import heatnet.test.test_util as test_util
import tensorflow as tf


class FileUtilTest(absltest.TestCase):
  """Tests for file utils."""

  def test_ext_to_local(self):
    """Tests ext to local."""
    with file_util.mkdtemp() as tmp_dir, file_util.mkdtemp() as tmp_dir2:
      data_path = os.path.join(tmp_dir, 'temp_data.nc')
      test_util.write_dummy_dataset(data_path, 't2m')
      self.assertEmpty(tf.io.gfile.listdir(tmp_dir2))
      file_util.ext_to_local(data_path, tmp_dir2)
      self.assertLen(tf.io.gfile.listdir(tmp_dir2), 1)
      os.remove(data_path)


if __name__ == '__main__':
  absltest.main()
