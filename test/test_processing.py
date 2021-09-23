"""Tests for functions and classes in data/processing.py."""
import glob
import os

from absl.testing import absltest
import heatnet.data.processing as hdp
import heatnet.file_util as file_util
import heatnet.test.test_util as test_util
import numpy as np
import xarray as xr


class CDSPreprocessorTest(absltest.TestCase):
  """Tests for CDSPreprocesor."""

  def test_init(self):
    """Tests CDSPreprocessor initialization."""
    with file_util.mkdtemp() as tmp_dir:
      data_paths = [
          os.path.join(tmp_dir, 'temp_data.nc'),
          os.path.join(tmp_dir, 'temp_data_2.nc')
      ]
      proc_path = os.path.join(tmp_dir, 'temp_proc_data.nc')
      variables = ['swvl1', 't2m']
      for path, var in zip(data_paths, variables):
        test_util.write_dummy_dataset(path, var)

      pp = hdp.CDSPreprocessor(data_paths, base_out_path=proc_path, mode='ext')
      self.assertEqual(pp.raw_files, data_paths)
      self.assertEqual(pp.base_out_path, proc_path)
      self.assertEqual(pp.lead_times, [1])
      self.assertEqual(pp.past_times, [0])
      pp.close()

      pp = hdp.CDSPreprocessor(
          data_paths[0], base_out_path=proc_path, mode='ext')
      self.assertEqual(pp.raw_files, data_paths[0])
      self.assertEqual(pp.base_out_path, proc_path)
      self.assertEqual(pp.lead_times, [1])
      self.assertEqual(pp.past_times, [0])
      pp.close()
      for path in data_paths:
        os.remove(path)

  def test_raw_to_batched_samples(self):
    """Tests default raw_to_batched_samples call."""
    tol = 1.0e-4
    with file_util.mkdtemp() as tmp_dir:
      path = os.path.join(tmp_dir, 'temp_data.nc')
      proc_path = os.path.join(tmp_dir, 'temp_proc_data.nc')
      proc_path1 = os.path.join(tmp_dir, 'temp_proc_data.000000.nc')
      test_util.write_dummy_dataset(path, 'swvl1')
      pp = hdp.CDSPreprocessor(path, base_out_path=proc_path, mode='ext')
      pp.raw_to_batched_samples()

      self.assertEqual(pp.pred_varlev_time, ['swvl1/0'])
      self.assertEqual(pp.tgt_varlev_time, ['swvl1/0/+1D'])
      with xr.open_dataset(path) as ds, xr.open_dataset(proc_path1) as proc_ds:
        self.assertTrue(
            np.allclose(
                ds.isel(time=0).swvl1.values,
                proc_ds.isel(sample=0).sel(
                    pred_varlev='swvl1/0').predictors.values,
                rtol=tol,
                atol=tol))
      os.remove(path)
      for f in glob.glob(os.path.join(tmp_dir, 'temp_proc*')):
        os.remove(f)
      pp.close()

  def test_offsets(self):
    """Tests correctness of time offsets from raw to processed data."""
    tol = 1.0e-4
    with file_util.mkdtemp() as tmp_dir:
      data_paths = [
          os.path.join(tmp_dir, 'temp_data.nc'),
          os.path.join(tmp_dir, 'temp_data_3.nc'),
          os.path.join(tmp_dir, 'temp_data_2.nc'),
      ]
      variables = ['t2m', 'swvl1', 't2m_anom']
      proc_path_1 = os.path.join(tmp_dir, 'temp_proc_data.000000.nc')
      for path, var in zip(data_paths, variables):
        test_util.write_dummy_dataset(path, var)

      pp = hdp.CDSPreprocessor(
          data_paths,
          past_times=[1, 2],
          lead_times=[1, 2],
          base_out_path=os.path.join(tmp_dir, 'temp_proc_data.nc'),
          mode='ext')
      pp.raw_to_batched_samples()

      with xr.open_dataset(proc_path_1) as proc_ds:
        with xr.open_dataset(data_paths[0]) as ds:
          # First possible target with lead time = 2
          raw_data_slice = (ds.isel(time=4).t2m.values)
          tgt_data_slice = (
              proc_ds.sel(tgt_varlev='t2m/0/+1D').isel(sample=1).targets.values)
          tgt2_data_slice = (
              proc_ds.sel(tgt_varlev='t2m/0/+2D').isel(sample=0).targets.values)
          pred0_data_slice = (
              proc_ds.sel(pred_varlev='t2m/0').isel(sample=2).predictors.values)
          pred1_data_slice = (
              proc_ds.sel(pred_varlev='t2m/0/-1D').isel(
                  sample=3).predictors.values)
          pred2_data_slice = (
              proc_ds.sel(pred_varlev='t2m/0/-2D').isel(
                  sample=4).predictors.values)
          self.assertTrue(
              np.allclose(raw_data_slice, tgt_data_slice, rtol=tol, atol=tol))
          self.assertTrue(
              np.allclose(raw_data_slice, tgt2_data_slice, rtol=tol, atol=tol))
          self.assertTrue(
              np.allclose(raw_data_slice, pred0_data_slice, rtol=tol, atol=tol))
          self.assertTrue(
              np.allclose(raw_data_slice, pred1_data_slice, rtol=tol, atol=tol))
          self.assertTrue(
              np.allclose(raw_data_slice, pred2_data_slice, rtol=tol, atol=tol))
          self.assertEqual(ds.time.values[2], proc_ds.sample.values[0])

        with xr.open_dataset(data_paths[2]) as ds:
          # First possible target with lead time = 2
          raw_data_slice = (ds.isel(time=4).t2m_anom.values)
          tgt_data_slice = (
              proc_ds.sel(tgt_varlev='t2m_anom/0/+1D').isel(
                  sample=1).targets.values)
          tgt2_data_slice = (
              proc_ds.sel(tgt_varlev='t2m_anom/0/+2D').isel(
                  sample=0).targets.values)
          pred0_data_slice = (
              proc_ds.sel(pred_varlev='t2m_anom/0').isel(
                  sample=2).predictors.values)
          pred1_data_slice = (
              proc_ds.sel(pred_varlev='t2m_anom/0/-1D').isel(
                  sample=3).predictors.values)
          pred2_data_slice = (
              proc_ds.sel(pred_varlev='t2m_anom/0/-2D').isel(
                  sample=4).predictors.values)
          self.assertTrue(
              np.allclose(raw_data_slice, tgt_data_slice, rtol=tol, atol=tol))
          self.assertTrue(
              np.allclose(raw_data_slice, tgt2_data_slice, rtol=tol, atol=tol))
          self.assertTrue(
              np.allclose(raw_data_slice, pred0_data_slice, rtol=tol, atol=tol))
          self.assertTrue(
              np.allclose(raw_data_slice, pred1_data_slice, rtol=tol, atol=tol))
          self.assertTrue(
              np.allclose(raw_data_slice, pred2_data_slice, rtol=tol, atol=tol))
      pp.close()
      for path in data_paths:
        os.remove(path)
      for f in glob.glob(os.path.join(tmp_dir, 'temp_proc*')):
        os.remove(f)

  def test_mean_std_recovery(self):
    """Tests recovery of dimensional data from processed normalized data."""
    tol = 1.0e-4
    with file_util.mkdtemp() as tmp_dir:
      data_paths = [
          os.path.join(tmp_dir, 'temp_data.nc'),
          os.path.join(tmp_dir, 'temp_data_3.nc'),
          os.path.join(tmp_dir, 'temp_data_2.nc'),
      ]
      variables = ['t2m', 'swvl1', 't2m_anom']
      proc_path_1 = os.path.join(tmp_dir, 'temp_proc_data.000000.nc')
      for path, var in zip(data_paths, variables):
        test_util.write_dummy_dataset(path, var)

      pp = hdp.CDSPreprocessor(
          data_paths,
          base_out_path=os.path.join(tmp_dir, 'temp_proc_data.nc'),
          past_times=[1, 2],
          lead_times=[1, 2],
          mode='ext')
      pp.raw_to_batched_samples(scale_variables=True)

      with xr.open_dataset(proc_path_1) as proc_ds:
        with xr.open_dataset(os.path.join(
            tmp_dir, 'temp_proc_data.scales.nc')) as scale_ds:
          with xr.open_dataset(data_paths[1]) as ds:
            raw_values = ds.isel(time=4).swvl1.values
            proc_values = proc_ds.isel(sample=2).sel(
                pred_varlev='swvl1/0').predictors.values
            proc_scaled_values = np.add(
                np.multiply(
                    proc_values,
                    scale_ds.sel(pred_varlev='swvl1/0').pred_std.values),
                scale_ds.sel(pred_varlev='swvl1/0').pred_mean.values)
            self.assertTrue(
                np.allclose(raw_values, proc_scaled_values, rtol=tol, atol=tol))
            proc_values = proc_ds.isel(sample=4).sel(
                pred_varlev='swvl1/0/-2D').predictors.values
            proc_scaled_values = np.add(
                np.multiply(
                    proc_values,
                    scale_ds.sel(pred_varlev='swvl1/0').pred_std.values),
                scale_ds.sel(pred_varlev='swvl1/0').pred_mean.values)
            self.assertTrue(
                np.allclose(raw_values, proc_scaled_values, rtol=tol, atol=tol))

          with xr.open_dataset(data_paths[2]) as ds:
            raw_values = ds.isel(time=4).t2m_anom.values
            proc_values = proc_ds.isel(sample=2).sel(
                pred_varlev='t2m_anom/0').predictors.values
            proc_scaled_values = np.add(
                np.multiply(
                    proc_values,
                    scale_ds.sel(pred_varlev='t2m_anom/0').pred_std.values),
                scale_ds.sel(pred_varlev='t2m_anom/0').pred_mean.values)
            self.assertTrue(
                np.allclose(raw_values, proc_scaled_values, rtol=tol, atol=tol))
            proc_values = proc_ds.isel(sample=3).sel(
                pred_varlev='t2m_anom/0/-1D').predictors.values
            proc_scaled_values = np.add(
                np.multiply(
                    proc_values,
                    scale_ds.sel(pred_varlev='t2m_anom/0').pred_std.values),
                scale_ds.sel(pred_varlev='t2m_anom/0').pred_mean.values)
            self.assertTrue(
                np.allclose(raw_values, proc_scaled_values, rtol=tol, atol=tol))

      pp.close()
      for path in data_paths:
        os.remove(path)
      for f in glob.glob(os.path.join(tmp_dir, 'temp_proc*')):
        os.remove(f)


if __name__ == '__main__':
  absltest.main()
