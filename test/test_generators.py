"""Tests for functions and classes in data/generators.py."""
import copy
import glob
import os

from absl.testing import absltest
import heatnet.data.generators as generators
import heatnet.data.processing as hdp
import heatnet.file_util as file_util
import heatnet.test.test_util as test_util
import numpy as np
import pandas as pd
import xarray as xr


class ShardedDataGeneratorTest(absltest.TestCase):
  """Tests for ShardedDataGenerator."""

  def test_init(self):
    """Tests initialization for different batch/file sharding configurations."""
    with file_util.mkdtemp() as tmp_dir:
      path = os.path.join(tmp_dir, 'temp_data.nc')
      proc_path = os.path.join(tmp_dir, 'temp_proc_data.nc')
      base_path = os.path.join(tmp_dir, 'temp_proc_data*')
      test_util.write_dummy_dataset(
          path, 'swvl1', date_range=('2016-12-31', '2017-12-31'))
      pp = hdp.CDSPreprocessor(path, base_out_path=proc_path, mode='ext')
      train_set = list(pd.date_range('2016-12-31', '2017-12-30', freq='1D'))

      # Test one batch = one file
      pp.raw_to_batched_samples()
      gen = generators.ShardedDataGenerator(train_set, base_path)
      self.assertLen(gen.filenames, len(train_set) // 32)
      self.assertEqual(gen.files_per_batch, gen.batches_per_file)
      for f in glob.glob(os.path.join(tmp_dir, 'temp_proc_data*')):
        os.remove(f)

      # Test one file = multiple batches
      pp.raw_to_batched_samples()
      gen = generators.ShardedDataGenerator(train_set, base_path, batch_size=8)
      self.assertLen(gen.filenames, len(train_set) // 32)
      self.assertEqual(gen.files_per_batch, 0)
      self.assertEqual(gen.batches_per_file, 4)
      for f in glob.glob(os.path.join(tmp_dir, 'temp_proc_data*')):
        os.remove(f)

      # Test one batch = multiple files
      pp.raw_to_batched_samples(batch_samples=8)
      gen = generators.ShardedDataGenerator(train_set, base_path)
      self.assertLen(
          gen.filenames,
          len(train_set) // 8 // gen.files_per_batch * gen.files_per_batch)
      self.assertEqual(gen.files_per_batch, 4)
      self.assertEqual(gen.batches_per_file, 0)
      for f in glob.glob(os.path.join(tmp_dir, 'temp_proc_data*')):
        os.remove(f)

  def test_shuffling(self):
    """Tests shuffling for different batch/file sharding configurations."""
    with file_util.mkdtemp() as tmp_dir:
      path = os.path.join(tmp_dir, 'temp_data.nc')
      proc_path = os.path.join(tmp_dir, 'temp_proc_data.nc')
      base_path = os.path.join(tmp_dir, 'temp_proc_data*')
      test_util.write_dummy_dataset(
          path, 'swvl1', date_range=('2016-12-31', '2017-12-31'))
      pp = hdp.CDSPreprocessor(path, base_out_path=proc_path, mode='ext')
      train_set = list(pd.date_range('2016-12-31', '2017-12-30', freq='1D'))

      # Test one file = multiple batches
      pp.raw_to_batched_samples()
      gen = generators.ShardedDataGenerator(
          train_set, base_path, batch_size=8, shuffle=True)
      batch_to_file_orig = copy.deepcopy(gen.batch_to_file)
      batch_to_samples_orig = copy.deepcopy(gen.batch_to_samples)
      # All files are pointed at by batches.
      self.assertEqual(set(gen.batch_to_file.values()), set(gen.filenames))
      self.assertTrue(
          np.allclose(batch_to_samples_orig[0], gen.batch_to_samples[0]))
      gen.on_epoch_end()
      # batch_to_file does not change since we shuffle through batch index.
      self.assertEqual(batch_to_file_orig[0], gen.batch_to_file[0])
      # batch_to_samples changes during shuffling.
      self.assertFalse(
          np.allclose(batch_to_samples_orig[0], gen.batch_to_samples[0]))
      for f in glob.glob(os.path.join(tmp_dir, 'temp_proc_data*')):
        os.remove(f)

      # Test one batch = multiple files
      pp.raw_to_batched_samples(batch_samples=8)
      gen = generators.ShardedDataGenerator(train_set, base_path, shuffle=True)
      batch_to_file_orig = copy.deepcopy(gen.batch_to_file)
      batch_to_samples_orig = copy.deepcopy(gen.batch_to_samples)
      # All files are pointed at by batches.
      self.assertEqual(
          set(sum(gen.batch_to_file.values(), [])), set(gen.filenames))
      gen.on_epoch_end()
      # batch_to_file changes during shuffling to randomize file groups.
      self.assertNotEqual(batch_to_file_orig[0], gen.batch_to_file[0])
      # batch_to_samples does not change since all file samples are used.
      self.assertTrue(
          np.allclose(batch_to_samples_orig[0], gen.batch_to_samples[0]))
      for f in glob.glob(os.path.join(tmp_dir, 'temp_proc_data*')):
        os.remove(f)


class FullDataGeneratorTest(absltest.TestCase):
  """Tests for FullDataGenerator."""

  def test_init(self):
    """Tests initialization for the FullDataGenerator."""
    with file_util.mkdtemp() as tmp_dir:
      path = os.path.join(tmp_dir, 'temp_data.nc')
      proc_path = os.path.join(tmp_dir, 'temp_proc_data.nc')
      file_path = os.path.join(tmp_dir, 'temp_proc_data.000000.nc')
      test_util.write_dummy_dataset(
          path, 'swvl1', date_range=('2016-12-31', '2017-12-31'))
      pp = hdp.CDSPreprocessor(path, base_out_path=proc_path, mode='ext')
      pp.raw_to_batched_samples(batch_samples=364)
      with xr.open_dataset(file_path) as ds:
        gen = generators.FullDataGenerator(ds)
        self.assertEqual(ds, gen.ds)
        self.assertEqual(ds.predictors.shape[1:], gen.input_shape)
        self.assertEqual(ds.targets.shape[1:], gen.output_shape)
      for f in glob.glob(os.path.join(tmp_dir, 'temp_proc_data*')):
        os.remove(f)


if __name__ == '__main__':
  absltest.main()
