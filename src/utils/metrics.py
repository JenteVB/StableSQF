# LOAD MODULES

# Standard library
from typing import List

# Proprietary

# Third party
import torch
import torch.nn.functional as F

def inexpolate_qf(quantile_levels: torch.Tensor,
                  quantile_forecast: torch.Tensor, 
                  quantile_forecast_levels: torch.Tensor,
                  device):
  '''
  Using interpolation and extrapolation to approximate quantile function.
  The quantile values are returned for a list of quantile_levels.
  The shape of iqf_quantile_level is batch_size x forecast_length x len(quantile_levels).
  '''
  quantile_levels = quantile_levels.to(device)
  quantile_forecast = quantile_forecast.to(device)
  quantile_forecast_levels = quantile_forecast_levels.to(device)
  iqf_quantile_level = torch.zeros(quantile_forecast.shape[0], quantile_forecast.shape[1], len(quantile_levels), device=device)
  for i_q, q in enumerate(quantile_levels):
    if quantile_forecast_levels[0] <= q <= quantile_forecast_levels[-1]:
      # select indices
      for i in range(len(quantile_forecast_levels) - 1):
        if quantile_forecast_levels[i] <= q <= quantile_forecast_levels[i + 1]:
          lower_index = i
          upper_index = i + 1
          break
      w_q = (quantile_forecast_levels[upper_index]-q)/(quantile_forecast_levels[upper_index]-quantile_forecast_levels[lower_index])      
      iqf_quantile_level[..., i_q] = w_q*quantile_forecast[..., lower_index]+(1-w_q)*quantile_forecast[..., upper_index]
    elif q < quantile_forecast_levels[0]:
      # exponential lower tail      
      epsilon = torch.finfo(quantile_forecast_levels.dtype).eps/2
      beta_l = torch.log((quantile_forecast_levels[1]+epsilon)/(quantile_forecast_levels[0]+epsilon)+epsilon)/(quantile_forecast[..., 1]-quantile_forecast[..., 0])
      iqf_quantile_level[..., i_q] = (1/beta_l)*torch.log(q/quantile_forecast_levels[1]) + quantile_forecast[..., 1]
      # or when quantile_forecast_levels[0] = 0.005 and 100 quantiles are used to approximate CRPS:
      # iqf_quantile_level[..., i_q] = quantile_forecast[..., 0]
    else: # if q > quantile_forecast_levels[-1]
      # exponential upper tail
      epsilon = torch.finfo(quantile_forecast_levels.dtype).eps/2      
      beta_r = torch.log((1-quantile_forecast_levels[-2]+epsilon)/(1-quantile_forecast_levels[-1]+epsilon)+epsilon)/(quantile_forecast[..., -1]-quantile_forecast[..., -2])
      iqf_quantile_level[..., i_q] = (1/beta_r)*torch.log((1-quantile_forecast_levels[-2])/(1-q)) + quantile_forecast[..., -2]
      # or when quantile_forecast_levels[-1] = 0.995 and 100 quantiles are used to approximate CRPS:
      # iqf_quantile_level[..., i_q] = quantile_forecast[..., -1]
  return iqf_quantile_level
  
def spline_qf(quantile_levels: torch.Tensor,
              gamma: torch.Tensor, 
              beta: torch.Tensor, 
              delta: torch.Tensor, 
              spline_pieces: int,
              device):
  '''
  Using linear isotonic regression splines to approximate quantile function.
  The quantile values are returned for a list of quantile_levels.
  The shape of sqf_quantile_level is batch_size x forecast_length x len(quantile_levels).
  '''
  quantile_levels = quantile_levels.to(device)
  gamma = gamma.to(device)
  beta = beta.to(device)
  delta = delta.to(device)
  sum_total = torch.zeros(gamma.shape[0], gamma.shape[1], len(quantile_levels), device=device)
  for i in range(1, spline_pieces):
    summand_p1 = torch.diff(beta, dim=-1)[..., i-1].unsqueeze(-1).expand(-1, -1, len(quantile_levels))
    summand_p2 = F.relu(quantile_levels.expand_as(sum_total)-
                        torch.sum(delta[..., :i], -1).unsqueeze(-1).expand(-1, -1, len(quantile_levels)))
    summand = summand_p1 * summand_p2
    sum_total = sum_total + summand
  # Evaluate spline quantile fuction
  sqf_quantile_level = (gamma.expand(-1, -1, len(quantile_levels)) + # intercept 
                        beta[..., 0].unsqueeze(-1).expand(-1, -1, len(quantile_levels)) * quantile_levels.expand_as(sum_total) + # l = 0
                        sum_total) # sum_{l=1}^{L-1}
  return sqf_quantile_level

def spline_qf_fixed_knots(quantile_levels: torch.Tensor,
                          gamma: torch.Tensor, 
                          beta: torch.Tensor, 
                          quantile_levels_knots: torch.Tensor, 
                          spline_pieces: int,
                          device):
  '''
  Using linear isotonic regression splines to approximate quantile function.
  The quantile values are returned for a list of quantile_levels.
  The shape of sqf_quantile_level is batch_size x forecast_length x len(quantile_levels).
  '''
  quantile_levels = quantile_levels.to(device)
  gamma = gamma.to(device)
  beta = beta.to(device)
  quantile_levels_knots = quantile_levels_knots.to(device)
  sum_total = torch.zeros(gamma.shape[0], gamma.shape[1], len(quantile_levels), device=device)
  for i in range(1, spline_pieces):
    summand_p1 = torch.diff(beta, dim=-1)[..., i-1].unsqueeze(-1).expand(-1, -1, len(quantile_levels))
    summand_p2 = F.relu(quantile_levels.expand_as(sum_total)-quantile_levels_knots[i-1].expand_as(sum_total))
    summand = summand_p1 * summand_p2
    sum_total = sum_total + summand
  # Evaluate spline quantile fuction
  sqf_quantile_level = (gamma.expand(-1, -1, len(quantile_levels)) + # intercept 
                        beta[..., 0].unsqueeze(-1).expand(-1, -1, len(quantile_levels)) * quantile_levels.expand_as(sum_total) + # l = 0
                        sum_total) # sum_{l=1}^{L-1}
  return sqf_quantile_level

def quantile_score(quantile_levels: torch.Tensor,
                   quantile_forecasts: torch.Tensor,
                   actuals: torch.Tensor,
                   device):
  '''
  Calculate the quantile score for a quantile forecast for a specific quantile level.
  The quantile scores are returned for a list of quantile_levels.
  The shape of qs_quantile_level is batch_size x forecast_length x len(quantile_levels).
  '''
  quantile_levels = quantile_levels.to(device)
  quantile_forecasts = quantile_forecasts.to(device)
  actuals = actuals.to(device)
  qs_p1 = (quantile_levels.expand_as(quantile_forecasts) * 
          F.relu(actuals.unsqueeze(-1).expand(-1, -1, len(quantile_levels)) - quantile_forecasts))
  qs_p2 = ((1 - quantile_levels.expand_as(quantile_forecasts)) * 
          F.relu(quantile_forecasts -  actuals.unsqueeze(-1).expand(-1, -1, len(quantile_levels))))
  qs_quantile_level = 2 * (qs_p1 + qs_p2)
  return qs_quantile_level

def log_score(quantile_forecasts: torch.Tensor, # shape = batch_size x forecast_length x n_quantile_levels
              actuals: torch.Tensor, # shape = batch_size x forecast_length
              mean_n_obs_in_bin: int = 5,
              max_log_score: int = 20):
  '''
  Calculate the log score for a specific observation based on quantile forecasts for equidistant quantile levels.
  The relative frequencies of an empirical histogram constructed based on the quantile forecasts are used to approximate the logscore.
  The shape of log_scores is batch_size x forecast_length.
  '''
  log_scores = torch.empty_like(actuals, device="cpu")
  n_bins = int(quantile_forecasts.shape[2]/mean_n_obs_in_bin)
  for sample, quantile_forecasts_sample in enumerate(quantile_forecasts):
    for horizon, quantile_forecasts_sample_horizon in enumerate(quantile_forecasts_sample):
      density_values, bin_edges = torch.histogram(quantile_forecasts_sample_horizon.cpu(), n_bins, density=True)
      actual = actuals[sample, horizon].cpu()
      if (actual < torch.min(bin_edges)) or (actual > torch.max(bin_edges)):
          log_score_sample_horizon = max_log_score
      else:
          dv = torch.mean(density_values[(bin_edges[:-1]<=actual)&(bin_edges[1:]>=actual)&(density_values>0)])
          if torch.isnan(dv):
            dv = torch.min(density_values[density_values>0])
          log_score_sample_horizon = -torch.log(dv)
      log_scores[sample, horizon] = log_score_sample_horizon
  return log_scores

# # Test log_score
# # Inputs
# quantile_forecasts = torch.rand(2, 3, 100)
# quantile_forecasts, _ = torch.sort(quantile_forecasts)
# quantile_forecasts.shape
# mean_n_obs_in_bin = 5
# max_log_score = 20
# # Selection and histogram calculation
# n_bins = int(quantile_forecasts.shape[2]/mean_n_obs_in_bin)
# sample = 0
# quantile_forecasts_sample = quantile_forecasts[sample]
# horizon = 0
# quantile_forecasts_sample_horizon = quantile_forecasts_sample[horizon]
# density_values, bin_edges = torch.histogram(quantile_forecasts_sample_horizon, n_bins, density=True)
# print(density_values)
# print(bin_edges)
# # Variables to manipulate
# density_values[0] = density_values[0] + density_values[1]
# density_values[1] = 0
# density_values.sum()
# actual = torch.full((1,), 0.1)
# actual = bin_edges[1]
# # Scenario 1
# ((actual < torch.min(bin_edges)) or (actual > torch.max(bin_edges)))
# log_score_sample_horizon = max_log_score
# # Scenario 2
# dv = torch.mean(density_values[(bin_edges[:-1]<=actual)&(bin_edges[1:]>=actual)&(density_values>0)])
# # Scenario 2b
# torch.isnan(dv)
# dv = torch.min(density_values[density_values>0])
# # Output
# log_score_sample_horizon = -torch.log(dv)
