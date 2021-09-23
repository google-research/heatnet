"""Tests for functions and classes in evaluation/util.py."""
import glob
import os

from absl.testing import absltest
import heatnet.data.generators as generators
import heatnet.evaluation.util as eval_util
import heatnet.file_util as file_util
import heatnet.model
import heatnet.model.estimators as estimators
import heatnet.test.test_util as test_util
import numpy as np
import pandas as pd
import xarray as xr


class MetricEvalTest(absltest.TestCase):
  """Tests for forecast_metric method."""

  def test_eval_global_error(self):
    """Tests correct metric output structure for global error evaluation."""
    eval_set = list(pd.date_range('2016-12-31', '2017-6-29', freq='1D'))
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
          eval_set, base_path, mode='ext', batch_size=32)
      estimator = estimators.ShardedDataEstimator(
          model, gen, include_targets=True)
      forecast = estimator.predict()

      metrics = eval_util.forecast_metric(forecast, target_var='swvl1')[0]
      self.assertIsInstance(metrics, xr.Dataset)
      self.assertNotIn('face', metrics.dims)
      self.assertNotIn('height', metrics.dims)
      self.assertNotIn('width', metrics.dims)
      self.assertIn('time', metrics.dims)
      self.assertIn('tgt_varlev', metrics.dims)
      self.assertIn('swvl1', metrics.forecast_mse.tgt_varlev)

      # All positive random fields, so all values are above or equal to 0
      same_metrics = eval_util.forecast_metric(
          forecast, target_var='swvl1', mask_below=0.0)[0]
      self.assertTrue(
          np.allclose(same_metrics.forecast_mse.values,
                      metrics.forecast_mse.values))

      diff_metrics = eval_util.forecast_metric(
          forecast, target_var='swvl1', mask_below=0.5, mask_above=0.6)[0]
      self.assertFalse(
          np.allclose(diff_metrics.forecast_mse.values,
                      metrics.forecast_mse.values))

      for f in glob.glob(base_path):
        os.remove(f)


if __name__ == '__main__':
  absltest.main()
