"""Tests for functions in data/util.py."""
from absl.testing import absltest
import heatnet.data.util as data_util


class DataUtilTest(absltest.TestCase):
  """Tests for data utils."""

  def test_coord_and_prefix(self):
    """Tests correct coordinate and prefix outputs for all variable types."""
    self.assertEqual(
        data_util.coord_and_prefix('predictors'), ('pred_varlev', 'pred'))
    self.assertEqual(
        data_util.coord_and_prefix('targets'), ('tgt_varlev', 'tgt'))
    with self.assertRaises(ValueError):
      data_util.coord_and_prefix('foo')

  def test_varlev_pairs(self):
    """Tests correct definition of variable/level pairs."""
    variables = ['ciwc', 'd', 'z', 'z', 'z']
    levels = [500, 500, 300, 500, 700]
    self.assertEqual(
        data_util.get_varlev_pairs(variables, levels),
        ['ciwc/500', 'd/500', 'z/300', 'z/500', 'z/700'])
    variables = ['ciwc', 'd', 'z', 'z', 'z']
    levels = [300, 500, 700]
    with self.assertRaises(ValueError):
      data_util.get_varlev_pairs(variables, levels)


if __name__ == '__main__':
  absltest.main()
