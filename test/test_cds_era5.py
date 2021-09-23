"""Tests for functions and classes in data/cds_era5.py."""
import os

from absl.testing import absltest
import heatnet.data as hd
import heatnet.file_util as file_util
import heatnet.test.test_util as test_util
import numpy as np
import xarray as xr


class CDSHandlerTest(absltest.TestCase):
  """Tests for CDSHandler."""

  def test_init_pres_var(self):
    """Tests variable and level definitions for pressure level variable."""
    variables = ['geopotential']
    levels = [500]
    with file_util.mkdtemp() as tmp_dir:
      era = hd.CDSHandler(root_directory=tmp_dir, file_id='era')
      self.assertEmpty(era.dataset_variables)
      self.assertEmpty(era.dataset_levels)
      era.set_variables(variables)
      era.set_levels(levels)
      self.assertEqual(era.dataset_variables, variables)
      self.assertEqual(era.dataset_levels, levels)

  def test_init_surf_var(self):
    """Tests variable and level definitions for surface variable."""
    variables = ['2m_temperature']
    levels = []
    with file_util.mkdtemp() as tmp_dir:
      era = hd.CDSHandler(root_directory=tmp_dir, file_id='era')
      self.assertEmpty(era.dataset_variables)
      self.assertEmpty(era.dataset_levels)
      era.set_variables(variables)
      era.set_levels(levels)
      self.assertEqual(era.dataset_variables, variables)
      self.assertEmpty(era.dataset_levels)

  def test_sorted(self):
    """Tests variable and level sorting."""
    variables = ['temperature', 'geopotential']
    levels = [1000, 500]
    with file_util.mkdtemp() as tmp_dir:
      era = hd.CDSHandler(root_directory=tmp_dir, file_id='era')
      era.set_variables(variables)
      era.set_levels(levels)
      self.assertEqual(era.dataset_variables, ['geopotential', 'temperature'])
      self.assertEqual(era.dataset_levels, [500, 1000])

  def test_variables(self):
    """Tests variables are ERA5 compliant."""
    variables = ['foo']
    with file_util.mkdtemp() as tmp_dir:
      era = hd.CDSHandler(root_directory=tmp_dir, file_id='era')
      with self.assertRaises(AssertionError):
        era.set_variables(variables)

  def test_resampling(self):
    """Tests resampled datasets."""
    freq = '3D'
    tol = 1.0e-4
    variables = ['2m_temperature', 'geopotential']
    level = 500
    with file_util.mkdtemp() as tmp_dir:
      # Create raw datasets
      raw_paths = [
          os.path.join(tmp_dir, 'era_2m_temperature.nc'),
          os.path.join(tmp_dir, 'era_geopotential_500.nc')
      ]
      rsmp_paths = [
          '.'.join((path.split('.')[0], 'rsmp', path.split('.')[1]))
          for path in raw_paths
      ]
      for var, path in zip(variables, raw_paths):
        test_util.write_dummy_dataset(path, hd.get_short_name(var))

      # Initialize CDSHandler, set vars, levels and raw_files.
      era = hd.CDSHandler(root_directory=tmp_dir, file_id='era')
      era.set_variables(variables)
      era.set_levels([level])
      era.open()
      # Verify paths
      self.assertEqual(era.raw_files, raw_paths)

      era.resample_files(freq)
      # Verify updated paths pointing at resampled files.
      self.assertEqual(era.raw_files, rsmp_paths)

      # Verify statistics of resample files
      with xr.open_mfdataset(raw_paths) as ds, xr.open_mfdataset(
          rsmp_paths) as r_ds:
        self.assertTrue(
            np.allclose(
                ds.t2m.mean('time').values,
                r_ds.t2m.mean('time').values,
                rtol=tol,
                atol=tol))
      for (raw_file, rsmp_file) in zip(raw_paths, rsmp_paths):
        os.remove(raw_file)
        os.remove(rsmp_file)

  def test_climat_anom(self):
    """Tests computation of climatological anomalies."""
    tol = 1.0e-4
    variables = ['geopotential', '2m_temperature']
    level = 500
    date_range = ('2013-12-31', '2017-12-31')
    with file_util.mkdtemp() as tmp_dir:
      path = os.path.join(tmp_dir, 'era_geopotential_' + str(level) + '.nc')
      test_util.write_dummy_dataset(
          path, hd.get_short_name('geopotential'), date_range=date_range)
      path = os.path.join(tmp_dir, 'era_2m_temperature.nc')
      test_util.write_dummy_dataset(
          path, hd.get_short_name('2m_temperature'), date_range=date_range)
      era = hd.CDSHandler(root_directory=tmp_dir, file_id='era')
      era.set_variables(variables)
      era.set_levels([level])
      era.get_climatology_anomalies(variables, [level])
      # Restore filenames
      era.open()
      era.close()
      with xr.open_mfdataset(era.raw_files) as ds:
        # Global
        self.assertTrue(
            np.allclose(ds.z_anom.mean().values, 0, rtol=tol, atol=tol))
        self.assertTrue(
            np.allclose(ds.z_anom.std().values, 1, rtol=tol, atol=tol))
        self.assertTrue(
            np.allclose(ds.t2m_anom.mean().values, 0, rtol=tol, atol=tol))
        self.assertTrue(
            np.allclose(ds.t2m_anom.std().values, 1, rtol=tol, atol=tol))
        # Local
        self.assertTrue(
            np.allclose(ds.z_anom.mean('time').values, 0, rtol=tol, atol=tol))
        self.assertTrue(
            np.allclose(ds.z_anom.std('time').values, 1, rtol=tol, atol=tol))
        self.assertTrue(
            np.allclose(ds.t2m_anom.mean('time').values, 0, rtol=tol, atol=tol))
        self.assertTrue(
            np.allclose(ds.t2m_anom.std('time').values, 1, rtol=tol, atol=tol))
      for file in era.raw_files:
        os.remove(file)


if __name__ == '__main__':
  absltest.main()
