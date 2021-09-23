# Copyright 2021 Google LLC
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
"""Utils for evaluating keras HeatNet models."""
from typing import Optional, Tuple, List, Union

import numpy as np
import xarray as xr


def mse(
    true_target: xr.DataArray,
    estimate: xr.DataArray,
    mask: Optional[xr.DataArray],
    average_dims: Tuple[str, ...] = ('face', 'height', 'width')
) -> xr.DataArray:
  """Evaluates the mean squared error between two data arrays.

  Args:
    true_target: DataArray containing true target values.
    estimate: DataArray containing an estimate of the target.
    mask: Mask over which the result is computed.
    average_dims: Dimensions over which the result is averaged.

  Returns:
    A DataArray containing the evaluated mean squared error.
  """
  metric_da = np.square(np.subtract(estimate, true_target))
  if mask is not None:
    metric_da = metric_da.where(mask)
  return metric_da.mean(average_dims)


def anom_corr(
    true_target: xr.DataArray,
    estimate: xr.DataArray,
    mask: Optional[xr.DataArray],
    average_dims: Tuple[str, ...] = ('face', 'height', 'width')
) -> xr.DataArray:
  """Evaluates the anomaly correlation coefficient between two data arrays.

  The definition of the anomaly correlation coefficient follows the centered
  version in Eq. 1 of Wulff & Domeisen, 2019
  (https://doi.org/10.1029/2019GL084314). It is assumed that the given fields
  are already non-normalized climatological anomalies.

  Args:
    true_target: DataArray containing true target climatological anomalies.
    estimate: DataArray containing an estimate of the target climatological
      anomalies.
    mask: Mask over which the result is computed.
    average_dims: Dimensions over which the result is averaged.

  Returns:
    A DataArray containing the evaluated anomaly correlation coefficient.
  """
  true_anom = true_target.where(mask) - true_target.where(mask).mean(
      average_dims)
  estimate_anom = estimate.where(mask) - estimate.where(mask).mean(average_dims)

  return np.divide(
      np.multiply(true_anom, estimate_anom).sum(average_dims),
      np.sqrt(
          np.multiply(
              np.square(true_anom).sum(average_dims),
              np.square(estimate_anom).sum(average_dims))))


def _eval_metric(
    metric: str,
    true_target: xr.DataArray,
    estimate: xr.DataArray,
    mask: Optional[xr.DataArray],
    average_dims: Tuple[str, ...] = ('face', 'height', 'width')
) -> xr.DataArray:
  """Evaluates the requested metric between two data arrays.

  Args:
    metric: Metric to be evaluated.
    true_target: DataArray containing true target values.
    estimate: DataArray containing an estimate of the target.
    mask: Mask over which the result is computed.
    average_dims: Dimensions over which the result is averaged.

  Returns:
    A DataArray containing the evaluated metric.

  Raises:
    NotImplementedError: If the requested metric has not been implemented.
  """
  if metric == 'mse':
    return mse(true_target, estimate, mask, average_dims)
  elif metric == 'rmse':
    return np.sqrt(mse(true_target, estimate, mask, average_dims))
  elif metric == 'anom_corr':
    return anom_corr(true_target, estimate, mask, average_dims)
  else:
    raise NotImplementedError(f'Metric {metric} has not been implemented.')


def get_seasonal_mask(targets: xr.DataArray, predictors: xr.Dataset,
                      season: str) -> xr.DataArray:
  """Returns a seasonal mask over the targets DataArray for metric evaluation.

  Args:
    targets: DataArray containing targets.
    predictors: Dataset containing predictor fields.
    season: Season to set as true in the boolean mask, may be summer or winter.

  Returns:
    A boolean mask over the target fields.

  Raises:
    NotImplementedError: If the given season is not summer or winter.
  """
  if season == 'summer':
    nh_months = 'JJA'
    sh_months = 'DJF'
  elif season == 'winter':
    nh_months = 'DJF'
    sh_months = 'JJA'
  else:
    raise NotImplementedError('The only implemented seasons are summer and'
                              f'winter, but {season} was passed,')

  lat = predictors.sel(pred_varlev='sin_latitude').predictors.values[:,
                                                                     np.newaxis]
  nh_mask = np.multiply(
      lat > 0,
      np.expand_dims(targets.time.dt.season == nh_months, axis=[1, 2, 3, 4]))
  sh_mask = np.multiply(
      lat < 0,
      np.expand_dims(targets.time.dt.season == sh_months, axis=[1, 2, 3, 4]))
  return nh_mask | sh_mask


def get_boolean_mask(
    targets: xr.DataArray,
    predictors: Optional[xr.Dataset] = None,
    mask_below: Optional[float] = None,
    mask_above: Optional[float] = None,
    only_summer: bool = False,
    only_winter: bool = False,
    only_land: bool = False) -> Union[xr.DataArray, np.ndarray, bool]:
  """Returns a boolean mask over the targets DataArray for metric evaluation.

  Args:
    targets: DataArray containing targets.
    predictors: Dataset containing predictor fields.
    mask_below: Masks values below threshold from metric evaluation, if not
      None.
    mask_above: Masks values above threshold from metric evaluation, if not
      None.
    only_summer: If True, aggregates results only over summer hemispheres,
      defined as 'JJA' for the Northern Hemisphere and 'DJF' for the Southern
      Hemisphere. This option requires passing a predictors.
    only_winter: If True, aggregates results only over winter hemispheres,
      defined as 'DJF' for the Northern Hemisphere and 'JJA' for the Southern
      Hemisphere. This option requires passing a predictors.
    only_land: If True, aggregates results only over land. This option requires
      passing a predictors with a land-sea mask.

  Returns:
    A boolean mask over the target fields.

  Raises:
    ValueError: If the requested masks are inconsistent.
  """
  if only_summer and only_winter:
    raise ValueError('True values of only_summer and only_winter are'
                     'mutually exclusive.')

  if (only_summer or only_winter or only_land) and predictors is None:
    raise ValueError('Season restricted metrics and location restricted metrics'
                     'require passing a predictor dataset.')
  mask = True
  if mask_above is not None:
    mask = np.multiply(mask, targets < mask_above)
  if mask_below is not None:
    mask = np.multiply(mask, targets > mask_below)
  if only_summer:
    mask = np.multiply(mask,
                       get_seasonal_mask(targets, predictors, season='summer'))
  elif only_winter:
    mask = np.multiply(mask,
                       get_seasonal_mask(targets, predictors, season='winter'))
  if only_land and predictors is not None:
    land = predictors.sel(pred_varlev='lsm').predictors.values[:, np.newaxis]
    mask = np.multiply(mask, land > 0)

  return mask


def forecast_metric(forecast_ds: xr.Dataset,
                    metric: str = 'mse',
                    predictors_ds: Optional[xr.Dataset] = None,
                    climatology_metric: bool = False,
                    average_dims: Tuple[str, ...] = ('face', 'height', 'width'),
                    target_var: str = 't2m_anom/0',
                    mask_below: Optional[float] = None,
                    mask_above: Optional[float] = None,
                    only_summer: bool = False,
                    only_winter: bool = False,
                    only_land: bool = False) -> List[xr.Dataset]:
  """Returns aggregate metric evaluations given a forecast and a target.

  If the predictors are passed as an input, the method also returns the
  evaluation of a persistence model. If climatology_metric is True, the
  metrics are also returned for a climatology model. Note that this option
  assumes that the targets are climatology anomalies, which means that the
  climatology model would return tensors of zeroes for any metric considered.

  Args:
    forecast_ds: Dataset containing forecast and targets.
    metric: Metric to be evaluated.
    predictors_ds: Dataset containing predictor fields. If given, returns the
      metric evaluation for a persistence model.
    climatology_metric: If True, return the metric for a climatology model.
    average_dims: Tuple of dimensions to average and squash.
    target_var: The target variable of the model, without lead time suffix.
    mask_below: Masks values below threshold from metric evaluation, if not
      None.
    mask_above: Masks values above threshold from metric evaluation, if not
      None.
    only_summer: If True, aggregates results only over summer hemispheres,
      defined as 'JJA' for the Northern Hemisphere and 'DJF' for the Southern
      Hemisphere. This option requires passing a predictors_ds.
    only_winter: If True, aggregates results only over winter hemispheres,
      defined as 'DJF' for the Northern Hemisphere and 'JJA' for the Southern
      Hemisphere. This option requires passing a predictors_ds.
    only_land: If True, aggregates results only over land. This option requires
      passing a predictors_ds with a land-sea mask.

  Returns:
    A list of datasets containing the aggregate metric evaluations.
  """
  forecast = forecast_ds.forecast
  target = forecast_ds.targets
  # Compute persistence
  if predictors_ds is not None:
    persistence = np.repeat(
        predictors_ds.sel(pred_varlev=target_var).rename({
            'pred_varlev': 'tgt_varlev'
        }).predictors.values[:, np.newaxis],
        np.shape(target)[1],
        axis=1)
    persistence_da = xr.DataArray(
        persistence, coords=target.coords, dims=target.dims)

  mask = get_boolean_mask(
      target,
      predictors_ds,
      mask_below=mask_below,
      mask_above=mask_above,
      only_summer=only_summer,
      only_winter=only_winter,
      only_land=only_land)

  metric_da = _eval_metric(metric, target, forecast, mask,
                           average_dims).rename('_'.join(('forecast', metric)))
  metric_datasets = [metric_da.to_dataset()]

  if predictors_ds is not None:
    persistence_da = _eval_metric(metric, target, persistence_da, mask,
                                  average_dims).rename('_'.join(
                                      ('persistence', metric)))
    metric_datasets.append(persistence_da.to_dataset())

  if climatology_metric:
    climat_da = _eval_metric(metric, target, np.multiply(target, 0.0), mask,
                             average_dims).rename('_'.join(
                                 ('climatology', metric)))
    metric_datasets.append(climat_da.to_dataset())

  return metric_datasets
