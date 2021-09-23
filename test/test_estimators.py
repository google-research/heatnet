"""Tests for functions and classes in model/estimators.py."""
import glob
import os

from absl.testing import absltest
import heatnet.data.generators as generators
import heatnet.file_util as file_util
import heatnet.model
import heatnet.model.estimators as estimators
import heatnet.test.test_util as test_util
import pandas as pd
import xarray as xr


class FullDataEstimatorTest(absltest.TestCase):
  """Tests for FullDataEstimator."""

  def test_predict(self):
    """Tests correct forecast output structure."""
    inp_shape = (6, 48, 48, 1)
    model = heatnet.model.HeatUnet(inp_shape, inp_shape, 4).get_model()
    model.compile()
    with file_util.mkdtemp() as tmp_dir:
      path = os.path.join(tmp_dir, 'temp_data.nc')
      base_path = os.path.join(tmp_dir, 'temp_data*')
      test_util.write_cubed_sphere_dataset(
          path,
          'swvl1',
          date_range=('2016-12-31', '2017-6-30'),
          samples_per_file=64)
      with xr.open_mfdataset(glob.glob(base_path)) as ds:
        gen = generators.FullDataGenerator(ds)
        estimator = estimators.FullDataEstimator(model, gen)
        forecast = estimator.predict()

      self.assertIsInstance(forecast, xr.Dataset)
      self.assertIn('face', forecast.dims)
      self.assertIn('height', forecast.dims)
      self.assertIn('width', forecast.dims)
      self.assertIn('swvl1', forecast.forecast.tgt_varlev)
      for f in glob.glob(base_path):
        os.remove(f)


class ShardedDataEstimatorTest(absltest.TestCase):
  """Tests for ShardedDataEstimator."""

  def test_predict(self):
    """Tests correct forecast output structure."""
    train_set = list(pd.date_range('2016-12-31', '2017-6-29', freq='1D'))
    inp_shape = (6, 48, 48, 1)
    model = heatnet.model.HeatUnet(inp_shape, inp_shape, 4).get_model()
    model.compile()
    with file_util.mkdtemp() as tmp_dir:
      path = os.path.join(tmp_dir, 'temp_data.nc')
      base_path = os.path.join(tmp_dir, 'temp_data*')
      test_util.write_cubed_sphere_dataset(
          path,
          'swvl1',
          date_range=('2016-12-31', '2017-6-30'),
          samples_per_file=16)
      gen = generators.ShardedDataGenerator(
          train_set, base_path, mode='ext', batch_size=32)
      estimator = estimators.ShardedDataEstimator(model, gen)
      forecast = estimator.predict()

      self.assertIsInstance(forecast, xr.Dataset)
      self.assertIn('face', forecast.dims)
      self.assertIn('height', forecast.dims)
      self.assertIn('width', forecast.dims)
      self.assertIn('swvl1', forecast.forecast.tgt_varlev)
      for f in glob.glob(base_path):
        os.remove(f)


if __name__ == '__main__':
  absltest.main()
