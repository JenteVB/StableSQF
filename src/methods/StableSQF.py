# LOAD MODULES

# Standard library
from typing import Dict, Union, Literal

# Proprietary
from src.utils.metrics import (
    spline_qf_fixed_knots,
    quantile_score,
    log_score,
)

# Third party
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.parametrizations import weight_norm # Do not use as it results in problems when loading a model, instead use:
from torch.nn.utils import weight_norm
# from pytorch_forecasting.models import BaseModel
from lightning import LightningModule
from copy import deepcopy

def num_parameters(model):
    return sum(p.numel() for p in model.parameters())
    
class NBEATSQF_block(nn.Module):
    """
    This is the code for one N-BEATS-QF block.
    It outputs:
    (1) a forecast of the quantile function for the part of the input signal analyzed (one SQF for each time step in the forecast period):
    More specifically, for each time step i = 1,...,h, the block outputs the parameters of a spline quantile function (gamma_t+i, beta_t+i, delta_t+i).
    (2) and a backcast of the input to facilitate sequential analysis (feed residual to next block).
    For weight normalization see: https://arxiv.org/pdf/1602.07868.
    """
    def __init__(self,
                 backcast_length: int,
                 forecast_length: int,
                 hidden_layer_units: int,
                 spline_pieces: int):
        super().__init__()
        self.forecast_length = forecast_length
        self.spline_pieces = spline_pieces
        # Shared layers in block
        self.fc1 = nn.Linear(backcast_length, hidden_layer_units)
        self.fc2 = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc3 = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc4 = nn.Linear(hidden_layer_units, hidden_layer_units)
        # Task specific (backcast & forecast SQF parameters) layers in block
        self.fc_backcast = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc_gamma = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc_beta = nn.Linear(hidden_layer_units, hidden_layer_units)
        # Block output layers
        self.fc_backcast_output = nn.Linear(hidden_layer_units, backcast_length)   
        self.fc_gamma_outputs = nn.ModuleList()
        self.fc_beta_outputs = nn.ModuleList()
        for _ in range(forecast_length):
            gamma_output = nn.Linear(hidden_layer_units, 1)
            beta_output = nn.Linear(hidden_layer_units, spline_pieces)
            self.fc_gamma_outputs.append(gamma_output)
            self.fc_beta_outputs.append(beta_output)

    def forward(self, x: torch.Tensor):
        # Shared
        h1 = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope=0.01)
        h3 = F.leaky_relu(self.fc3(h2), negative_slope=0.01)
        h4 = F.leaky_relu(self.fc4(h3), negative_slope=0.01)
        # Task specific
        h_backcast = F.leaky_relu(self.fc_backcast(h4), negative_slope=0.01)
        h_gamma = F.leaky_relu(self.fc_gamma(h4), negative_slope=0.01)
        h_beta = F.leaky_relu(self.fc_beta(h4), negative_slope=0.01)
        # Outputs - backcast + SQF parameters per forecast period in forecast_length
        backcast = self.fc_backcast_output(h_backcast)
        gamma = torch.zeros((backcast.shape[0], # take batch_size from backcast
                             self.forecast_length,
                             1 # intercept of quantile function
                             ), 
                             dtype = torch.float,
                             device=x.device)
        beta = torch.zeros((backcast.shape[0], # take batch_size from backcast
                            self.forecast_length,
                            self.spline_pieces # slope parameters
                            ), 
                            dtype = torch.float,
                            device=x.device)
        for forecast_period in range(self.forecast_length):
            gamma[:, forecast_period, :] = self.fc_gamma_outputs[forecast_period](h_gamma)
            # betas should be positive to ensure monotonicity of the spline
            beta[:, forecast_period, :] = F.relu(self.fc_beta_outputs[forecast_period](h_beta))
        return backcast, gamma, beta
        
# # Test NBEATSQF_block
# network = NBEATSQF_block(5,2,10,4,True)
# x = torch.rand(20, 5) # batch_size = 20 x lookback_window_length = 5
# backcast, gamma, beta = network(x)
# print(backcast.shape) # dim = bs x bl
# print(gamma.shape) # dim = bs x fl x 1
# print(beta.shape) # dim = bs x fl x s_pieces
# print("Number of parameters:", num_parameters(network))
# (((5*10)+10)+((10*10)+10)+((10*10)+10)+((10*10)+10)+ # shared
#   3*((10*10)+10)+ # task-specific
#  ((10*5)+5)+ # backcast
#  2*(((10*1)+1)+((10*4)+4))) # sqf parameters per forecast period
#  # Difference is due to weight_norm

class NBEATSQF_module(nn.Module):
    """
    N-BEATS-QF outputs the parameters of the spline quantile functions for the different blocks in the network.
    The outputs are the parameters of the partial quantile forecasts and can be used to generate (a(ll)) quantile forecast(s) for each time step in the forecast period.
    """
    def __init__(self,
                 backcast_length: int,
                 forecast_length: int,
                 hidden_layer_units: int,
                 spline_pieces: int,
                 n_blocks: int,
                 n_blocks_shared: int):
        self.forecast_length = forecast_length
        self.spline_pieces = spline_pieces  
        super().__init__()
        # Init construction N-BEATS-QF blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = NBEATSQF_block(backcast_length,
                                   forecast_length,
                                   hidden_layer_units,
                                   spline_pieces)
            for _ in range(n_blocks_shared):
                self.blocks.append(block)

    def forward(self, backcast: torch.Tensor):
        # Containers for partial SQF parameters per forecast period in forecast_length
        gamma = torch.zeros((backcast.shape[0], # take batch_size from backcast
                            self.forecast_length, # quantile function per period in forecast_length
                            1, # intercept of quantile function
                            len(self.blocks)
                            ), 
                            dtype = torch.float,
                            device=backcast.device)
        beta = torch.zeros((backcast.shape[0], # take batch_size from backcast
                            self.forecast_length, # quantile function per period in forecast_length
                            self.spline_pieces, # slope parameters
                            len(self.blocks)
                            ), 
                            dtype = torch.float,
                            device=backcast.device)
        # Loop through blocks
        for block_id in range(len(self.blocks)):
            b, gamma_block, beta_block = self.blocks[block_id](backcast)
            backcast = backcast - b
            gamma[:, :, :, block_id] = gamma_block
            beta[:, :, :, block_id] = beta_block
        return gamma, beta

# # Test NBEATSQF_module
# network = NBEATSQF_module(5,2,10,4,3,False)
# x = torch.rand(20, 5) # batch_size = 20 x lookback_window_length = 5
# gamma, beta = network(x)
# print(gamma.shape) # dim = bs x fl x 1 x (n_blocks*n_blocks_shared)
# print(beta.shape) # dim = bs x fl x s_pieces x (n_blocks*n_blocks_shared)
# print("Number of parameters:", num_parameters(network)) 
# # [blocks] (1083*3)

class LitNBEATSQF(LightningModule):
    '''
    A LightningModule to operationalize NBEATSQF.
    '''
    def __init__(self,
                 # Model hypers
                 backcast_length_multiplier: int,
                 forecast_length: int,
                 quantile_levels_knots: torch.Tensor,
                 spline_pieces: int,
                 hidden_layer_units: int = 256,
                 wn: bool = False,
                 n_blocks: int = 10,
                 n_blocks_shared: int = 1,
                 ensemble_size: int = 1,
                 zero_mean: bool = True,
                 unit_variance: bool = True,                 
                 # Optim hypers
                 n_quantile_levels_training: int = 100,
                 n_quantile_levels_validation: int = 100,
                 n_quantile_levels_test: int = 100,
                 wc_quality: bool = False, # wCRPS center
                 wt_quality: bool = False, # wCRPS tails
                 loss_stability_type: Literal['W1', 'W2'] = 'W1', # Wasserstein distance
                 wc_stability: bool = False, # wWx center
                 wt_stability: bool = False, # wWx tails
                 lambda_stability: float = 0.0,
                 enforce_nonnegative_forecasts_metric_calculation: bool = False,
                 learning_rate: float = 1e-3,
                 explr_gamma: float = 1.00,
                 weight_decay: float = 0.00,
                 ema_decay: float = 0.00,
                 ):
        super().__init__()
        if wc_quality and wt_quality:
            raise ValueError("wc_quality and wt_quality cannot be set to True simultaneously")
        if (loss_stability_type != 'W1') and (loss_stability_type != 'W2'):
            raise ValueError("Invalid argument: loss_stability_type must be the string 'W1' or 'W2'")
        if wc_stability and wt_stability:
            raise ValueError("wc_stability and wt_stability cannot be set to True simultaneously")        
        if (not isinstance(lambda_stability, float)):
            raise ValueError("Invalid argument: lambda_stability must be a float")
        self.save_hyperparameters()
        self.model = nn.ModuleList([
            NBEATSQF_module(backcast_length=backcast_length_multiplier * forecast_length,
                            forecast_length=forecast_length,
                            hidden_layer_units=hidden_layer_units,
                            spline_pieces=spline_pieces,
                            n_blocks=n_blocks,
                            n_blocks_shared=n_blocks_shared,
                            ) for _ in range(ensemble_size)])
        self.ema_model = nn.ModuleList([
            NBEATSQF_module(backcast_length=backcast_length_multiplier * forecast_length,
                            forecast_length=forecast_length,
                            hidden_layer_units=hidden_layer_units,
                            spline_pieces=spline_pieces,
                            n_blocks=n_blocks,
                            n_blocks_shared=n_blocks_shared,
                            ) for _ in range(ensemble_size)])
        
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step using the given batch of data.
        For each input-output window in a batch, also a lagged input-output window is included in the batch 
        for stability calculations. The shape of lookback_windows and forecast_periods is batch_dim x backcast/forecast_length.
        """
        (
            _, _,
            loss_quality, _,
            loss_wc_quality, _,
            loss_wt_quality, _, 
            _,
            loss_stability1, _,
            loss_wc_stability1, _,
            loss_wt_stability1, _,
            loss_stability2, _,
            loss_wc_stability2, _,
            loss_wt_stability2, _,
            loss, w_quality, w_stability, 
            bs,
        ) = self._get_losses(batch,
                             self.hparams.n_quantile_levels_training,
                             self.hparams.wc_quality,
                             self.hparams.wt_quality,
                             self.hparams.loss_stability_type,
                             self.hparams.wc_stability,
                             self.hparams.wt_stability,
                             self.hparams.lambda_stability,
                             self.hparams.enforce_nonnegative_forecasts_metric_calculation,
                             False,
                             False)

        metrics = {"tloss_q": loss_quality,
                   "tloss_wcq": loss_wc_quality,
                   "tloss_wtq": loss_wt_quality,
                   "tloss_s1": loss_stability1,
                   "tloss_wcs1": loss_wc_stability1,
                   "tloss_wts1": loss_wt_stability1,
                   "tloss_s2": loss_stability2,
                   "tloss_wcs2": loss_wc_stability2,
                   "tloss_wts2": loss_wt_stability2,
                   "tloss": loss,
                   "w_quality": w_quality,
                   "w_stability": w_stability,
                   }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)

        return loss
        
    def validation_step(self, batch, batch_idx):
        """
        Performs validation using the given batch of data.
        For each input-output window in a batch, also a lagged input-output window is included in the batch 
        for stability calculations. The shape of lookback_windows and forecast_periods is batch_dim x backcast/forecast_length.
        """
        (
            rescaled_quantile_forecasts, _,
            loss_quality, _,
            loss_wc_quality, _,
            loss_wt_quality, _,
            _,
            loss_stability1, _,
            loss_wc_stability1, _,
            loss_wt_stability1, _,
            loss_stability2, _,
            loss_wc_stability2, _,
            loss_wt_stability2, _,
            loss, _, _,
            bs,
        ) = self._get_losses(batch,
                             self.hparams.n_quantile_levels_validation,
                             self.hparams.wc_quality,
                             self.hparams.wt_quality,
                             self.hparams.loss_stability_type,
                             self.hparams.wc_stability,
                             self.hparams.wt_stability,
                             self.hparams.lambda_stability,
                             self.hparams.enforce_nonnegative_forecasts_metric_calculation,
                             True,
                             False)
        
        metrics = {"vloss_q": loss_quality,
                   "vloss_wcq": loss_wc_quality,
                   "vloss_wtq": loss_wt_quality,
                   "vloss_s1": loss_stability1,
                   "vloss_wcs1": loss_wc_stability1,
                   "vloss_wts1": loss_wt_stability1,
                   "vloss_s2": loss_stability2,
                   "vloss_wcs2": loss_wc_stability2,
                   "vloss_wts2": loss_wt_stability2,
                   "vloss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)

        return rescaled_quantile_forecasts
    
    def test_step(self, batch, batch_idx):
        """
        Performs testing using the given batch of data.
        For each input-output window in a batch, also a lagged input-output window is included in the batch 
        for stability calculations. The shape of lookback_windows and forecast_periods is batch_dim x backcast/forecast_length.
        """
        (
            rescaled_quantile_forecasts, rescaled_quantile_forecasts_lagged,
            _, sCRPS,
            _, swcCRPS,
            _, swtCRPS,
            logS,
            _, sW1,
            _, swcW1,
            _, swtW1,
            _, sW2,
            _, swcW2,
            _, swtW2,
            _, _, _,
            bs,
        ) = self._get_losses(batch,
                             self.hparams.n_quantile_levels_test,
                             self.hparams.wc_quality,
                             self.hparams.wt_quality,
                             self.hparams.loss_stability_type,
                             self.hparams.wc_stability,
                             self.hparams.wt_stability,
                             self.hparams.lambda_stability,
                             self.hparams.enforce_nonnegative_forecasts_metric_calculation,
                             True,
                             True)
        
        metrics = {"sCRPS": sCRPS,
                   "swcCRPS": swcCRPS,
                   "swtCRPS": swtCRPS,
                   "logS": logS,
                   "sW1": sW1,
                   "swcW1": swcW1,
                   "swtW1": swtW1,
                   "sW2": sW2,
                   "swcW2": swcW2,
                   "swtW2": swtW2}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        
        rescaled_quantile_forecasts_all = torch.stack((rescaled_quantile_forecasts, rescaled_quantile_forecasts_lagged), dim=-1)

        return rescaled_quantile_forecasts_all

    def _get_losses(self,
                    batch,
                    n_quantile_levels,
                    wc_quality,
                    wt_quality,
                    loss_stability_type,
                    wc_stability,
                    wt_stability,
                    lambda_stability,
                    enforce_nonnegative_forecasts_metric_calculation,
                    evaluate,
                    calculate_metrics):
        x, _ = batch
        
        # Batch checks - handle batch_size = 1
        if len(x["encoder_cont"].shape) == 2:
            x["encoder_cont"] = x["encoder_cont"].unsqueeze(0)
        if len(x["decoder_cont"].shape) == 2:
            x["decoder_cont"] = x["decoder_cont"].unsqueeze(0)

        # if x["encoder_cont"].shape[0] != x["decoder_cont"].shape[0]:
        #     raise ValueError("Batch sizes of encoder and decoder do not match")
        bs = x["encoder_cont"].shape[0]

        # Batch data
        lookback_window = x["encoder_cont"][:,:,4] # shape = batch_size x lookback_window_length
        lookback_window_lagged = x["encoder_cont"][:,:,5] # shape = batch_size x lookback_window_length
        forecast_period = x["decoder_cont"][:,:,4] # shape = batch_size x forecast_length
        forecast_period_lagged = x["decoder_cont"][:,:,5] # shape = batch_size x forecast_length
        if evaluate:
            mean = x["decoder_cont"][:,:,0] # shape = batch_size x forecast_length
            std = x["decoder_cont"][:,:,1] # shape = batch_size x forecast_length
        
        # Data augementation on-the-fly
        if not evaluate:
            shift_value = torch.rand(bs, device=self.device)
            shift_sign = torch.randint(0, 2, (bs,), device=self.device) * 2 - 1
            shift = shift_sign * shift_value
            scale = torch.rand(bs, device=self.device) + 0.5
            lookback_window = (lookback_window + shift.unsqueeze(1)) * scale.unsqueeze(1)
            lookback_window_lagged = (lookback_window_lagged + shift.unsqueeze(1)) * scale.unsqueeze(1)
            forecast_period = (forecast_period + shift.unsqueeze(1)) * scale.unsqueeze(1)
            forecast_period_lagged = (forecast_period_lagged + shift.unsqueeze(1)) * scale.unsqueeze(1)
                        
        # Scaling factors
        if not calculate_metrics: # Scaling losses
            # Scaling constants for CRPS and W1 - shape = batch_size
            scaling_constant_abs_loss = torch.mean(torch.abs(torch.diff(lookback_window)), -1) + 1e-3 # for numerical stability
            scaling_constant_abs_lagged_loss = torch.mean(torch.abs(torch.diff(lookback_window_lagged)), -1) + 1e-3 # for numerical stability
            # scaling_constant_abs_loss = torch.ones(bs, device=self.device)
            # scaling_constant_abs_lagged_loss = torch.ones(bs, device=self.device)
            # Scaling constants For W2
            scaling_constant_sq_loss = torch.mean(torch.diff(lookback_window)**2, -1) + 1e-3 # for numerical stability
            # scaling_constant_sq_loss = torch.ones(bs, device=self.device)
        else: # Scaling metrics
            scaling_constant_abs = x["encoder_cont"][:,-1,2] # shape = batch_size
            scaling_constant_sq = x["encoder_cont"][:,-1,3] # shape = batch_size

        # Obtain parameters of the partial spline quantile functions
        ## Learnable parameters
        gamma = torch.zeros((bs, # batch_size
                             self.hparams.forecast_length, # quantile function per period in forecast_length
                             1, # intercept of quantile function
                             self.hparams.n_blocks*self.hparams.n_blocks_shared,
                             self.hparams.ensemble_size
                             ), 
                             dtype = torch.float,
                             device=self.device)
        gamma_lagged = torch.zeros_like(gamma)
        beta = torch.zeros((bs, # batch_size
                            self.hparams.forecast_length, # quantile function per period in forecast_length
                            self.hparams.spline_pieces, # slope parameters
                            self.hparams.n_blocks*self.hparams.n_blocks_shared,
                            self.hparams.ensemble_size
                            ), 
                            dtype = torch.float,
                            device=self.device)
        beta_lagged = torch.zeros_like(beta)
        
        for ensemble_id in range(self.hparams.ensemble_size):
            if not evaluate:
                gamma_id, beta_id = self.model[ensemble_id](lookback_window)
                gamma_lagged_id, beta_lagged_id = self.model[ensemble_id](lookback_window_lagged)
            else:
                with torch.no_grad():
                    gamma_id, beta_id = self.ema_model[ensemble_id](lookback_window)
                    gamma_lagged_id, beta_lagged_id = self.ema_model[ensemble_id](lookback_window_lagged)
            gamma[:, :, :, :, ensemble_id] = gamma_id
            beta[:, :, :, :, ensemble_id] = beta_id
            gamma_lagged[:, :, :, :, ensemble_id] = gamma_lagged_id
            beta_lagged[:, :, :, :, ensemble_id] = beta_lagged_id
        
        # Generate quantile predictions for CRPS approximation
        quantile_levels = torch.linspace((1/n_quantile_levels)/2, 
                                         1-((1/n_quantile_levels)/2), 
                                         n_quantile_levels,
                                         device=self.device)
        quantile_forecasts = torch.zeros((bs, # batch_size
                                          self.hparams.forecast_length,
                                          n_quantile_levels,
                                          self.hparams.n_blocks*self.hparams.n_blocks_shared,
                                          self.hparams.ensemble_size
                                          ), 
                                          dtype = torch.float,
                                          device=self.device)
        quantile_forecasts_lagged = torch.zeros_like(quantile_forecasts)
        for ensemble_id in range(self.hparams.ensemble_size):
            for block_id in range(self.hparams.n_blocks*self.hparams.n_blocks_shared):
                partial_quantile_forecasts = spline_qf_fixed_knots(quantile_levels,
                                                                   gamma[..., block_id, ensemble_id], 
                                                                   beta[..., block_id, ensemble_id],
                                                                   self.hparams.quantile_levels_knots,
                                                                   self.hparams.spline_pieces,
                                                                   self.device)
                quantile_forecasts[:, :, :, block_id, ensemble_id] = partial_quantile_forecasts
                partial_quantile_forecasts_lagged = spline_qf_fixed_knots(quantile_levels,
                                                                          gamma_lagged[..., block_id, ensemble_id], 
                                                                          beta_lagged[..., block_id, ensemble_id], 
                                                                          self.hparams.quantile_levels_knots,
                                                                          self.hparams.spline_pieces,
                                                                          self.device)
                quantile_forecasts_lagged[:, :, :, block_id, ensemble_id] = partial_quantile_forecasts_lagged
        quantile_forecasts = quantile_forecasts.sum(3).mean(-1) # sum over blocks and then take mean over ensemble_ids
        quantile_forecasts_lagged = quantile_forecasts_lagged.sum(3).mean(-1) # sum over blocks and then take mean over ensemble_ids

        rescaled_quantile_forecasts = 0
        rescaled_quantile_forecasts_lagged = 0
        if evaluate:
            # Reshape mean and std to match shape quantile_forecasts = batch_size x forecast_length x n_quantile_levels
            mean_reshaped = mean.unsqueeze(-1).expand(-1, -1, n_quantile_levels)
            std_reshaped = std.unsqueeze(-1).expand(-1, -1, n_quantile_levels)
            rescaled_quantile_forecasts = quantile_forecasts * std_reshaped + mean_reshaped
            if enforce_nonnegative_forecasts_metric_calculation:
                rescaled_quantile_forecasts = torch.clamp(rescaled_quantile_forecasts, 0)
            if calculate_metrics:
                rescaled_quantile_forecasts_lagged = quantile_forecasts_lagged * std_reshaped + mean_reshaped
                if enforce_nonnegative_forecasts_metric_calculation:
                    rescaled_quantile_forecasts_lagged = torch.clamp(rescaled_quantile_forecasts_lagged, 0)

        # Compute quantile scores for CRPS approximation
        if not calculate_metrics: # CRPS loss
            QS = quantile_score(quantile_levels, quantile_forecasts, forecast_period, self.device)
            QS_lagged = quantile_score(quantile_levels, quantile_forecasts_lagged, forecast_period_lagged, self.device)
        else: # CRPS metric
            rescaled_forecast_period = forecast_period * std + mean
            rescaled_QS = quantile_score(quantile_levels, rescaled_quantile_forecasts, rescaled_forecast_period, self.device)

        # Approximate CRPS variants
        loss_quality = 0
        loss_wc_quality = 0
        loss_wt_quality = 0
        sCRPS = 0
        swcCRPS = 0
        swtCRPS = 0
        if not calculate_metrics: # CRPS loss
            CRPS = torch.sum((1/n_quantile_levels)*QS, -1)
            CRPS_lagged = torch.sum((1/n_quantile_levels)*QS_lagged, -1)
            wcCRPS = torch.sum((1/n_quantile_levels)*QS*(quantile_levels*(1-quantile_levels)), -1)
            wcCRPS_lagged = torch.sum((1/n_quantile_levels)*QS_lagged*(quantile_levels*(1-quantile_levels)), -1)
            wtCRPS = torch.sum((1/n_quantile_levels)*QS*((2*quantile_levels-1)**2), -1)
            wtCRPS_lagged = torch.sum((1/n_quantile_levels)*QS_lagged*((2*quantile_levels-1)**2), -1)
            # Divide CRPS by scaling_constant_loss before taking mean across horizons and samples
            loss_quality = torch.mean(0.5 * (CRPS/scaling_constant_abs_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length) + 
                                             CRPS_lagged/scaling_constant_abs_lagged_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length)))
            loss_wc_quality = torch.mean(0.5 * (wcCRPS/scaling_constant_abs_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length) + 
                                                wcCRPS_lagged/scaling_constant_abs_lagged_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length)))
            loss_wt_quality = torch.mean(0.5 * (wtCRPS/scaling_constant_abs_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length) + 
                                                wtCRPS_lagged/scaling_constant_abs_lagged_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length)))
        else: # CRPS metric
            rescaled_CRPS = torch.sum((1/n_quantile_levels)*rescaled_QS, -1)
            rescaled_wcCRPS = torch.sum((1/n_quantile_levels)*rescaled_QS*(quantile_levels*(1-quantile_levels)), -1)
            rescaled_wtCRPS = torch.sum((1/n_quantile_levels)*rescaled_QS*((2*quantile_levels-1)**2), -1)
            # Divide rescaled_CRPS by scaling_constant before taking mean across horizons and samples
            sCRPS = torch.mean(rescaled_CRPS/scaling_constant_abs.unsqueeze(-1).expand(-1, self.hparams.forecast_length))
            swcCRPS = torch.mean(rescaled_wcCRPS/scaling_constant_abs.unsqueeze(-1).expand(-1, self.hparams.forecast_length))
            swtCRPS = torch.mean(rescaled_wtCRPS/scaling_constant_abs.unsqueeze(-1).expand(-1, self.hparams.forecast_length))

        # Approximate local scoring rule logS
        ## Only use as evaluation metric (not as loss) 
        ## - logS can be heavily influenced by slight deviations in the tail densities of forecasted distributions
        ## - loss should take into account the “distance” between the predicted distribution and the observed value 
        ##   and not just the assigned (local) probability for the observed value
        ## We do not consider weighted variants of logS as they are improper
        ## logS is scale-independent
        logS = 0
        if calculate_metrics:
            rescaled_logS = log_score(rescaled_quantile_forecasts, rescaled_forecast_period)
            logS = torch.mean(rescaled_logS)

        # Approximate Wasserstein-1 distance variants
        loss_stability1 = 0
        loss_wc_stability1 = 0
        loss_wt_stability1 = 0
        sW1 = 0
        swcW1 = 0
        swtW1 = 0
        if not calculate_metrics: # W1 loss
            W1 = torch.sum(torch.abs(quantile_forecasts[:,:-1,:]-
                                     quantile_forecasts_lagged[:,1:,:])*
                           (1/n_quantile_levels), -1)
            wcW1 = torch.sum(torch.abs(quantile_forecasts[:,:-1,:]-
                                       quantile_forecasts_lagged[:,1:,:])*
                             (quantile_levels*(1-quantile_levels))*
                             (1/n_quantile_levels), -1)
            wtW1 = torch.sum(torch.abs(quantile_forecasts[:,:-1,:]-
                                       quantile_forecasts_lagged[:,1:,:])*
                             ((2*quantile_levels-1)**2)*
                             (1/n_quantile_levels), -1)
            # Divide W1 by scaling_constant_loss before taking mean across horizons and samples
            loss_stability1 = torch.mean(W1/scaling_constant_abs_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1))
            loss_wc_stability1 = torch.mean(wcW1/scaling_constant_abs_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1))
            loss_wt_stability1 = torch.mean(wtW1/scaling_constant_abs_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1))
        else: # W1 metric
            rescaled_W1 = torch.sum(torch.abs(rescaled_quantile_forecasts[:,:-1,:]-
                                              rescaled_quantile_forecasts_lagged[:,1:,:])*
                                    (1/n_quantile_levels), -1)
            rescaled_wcW1 = torch.sum(torch.abs(rescaled_quantile_forecasts[:,:-1,:]-
                                                rescaled_quantile_forecasts_lagged[:,1:,:])*
                                      (quantile_levels*(1-quantile_levels))*
                                      (1/n_quantile_levels), -1)
            rescaled_wtW1 = torch.sum(torch.abs(rescaled_quantile_forecasts[:,:-1,:]-
                                                rescaled_quantile_forecasts_lagged[:,1:,:])*
                                      ((2*quantile_levels-1)**2)*
                                      (1/n_quantile_levels), -1)
            # Divide rescaled_W1 by scaling_constant before taking mean across horizons and samples 
            sW1 = torch.mean(rescaled_W1/scaling_constant_abs.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1))
            swcW1 = torch.mean(rescaled_wcW1/scaling_constant_abs.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1))
            swtW1 = torch.mean(rescaled_wtW1/scaling_constant_abs.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1))

        # Approximate Wasserstein-2 distance variants 
        loss_stability2 = 0
        loss_wc_stability2 = 0
        loss_wt_stability2 = 0
        sW2 = 0
        swcW2 = 0
        swtW2 = 0
        if not calculate_metrics: # W2 loss
            W2_sq = torch.sum(((quantile_forecasts[:,:-1,:]-
                                quantile_forecasts_lagged[:,1:,:])**2)*
                              (1/n_quantile_levels), -1)
            wcW2_sq = torch.sum(((quantile_forecasts[:,:-1,:]-
                                  quantile_forecasts_lagged[:,1:,:])**2)*
                                (quantile_levels*(1-quantile_levels))*
                                (1/n_quantile_levels), -1)
            wtW2_sq = torch.sum(((quantile_forecasts[:,:-1,:]-
                                  quantile_forecasts_lagged[:,1:,:])**2)*
                                ((2*quantile_levels-1)**2)*
                                (1/n_quantile_levels), -1)
            # Divide W2_sq by scaling_constant_loss before taking sqrt and mean across horizons and samples
            loss_stability2 = torch.mean(torch.sqrt(W2_sq/scaling_constant_sq_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1)))
            loss_wc_stability2 = torch.mean(torch.sqrt(wcW2_sq/scaling_constant_sq_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1)))
            loss_wt_stability2 = torch.mean(torch.sqrt(wtW2_sq/scaling_constant_sq_loss.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1)))
        else: # W2 metric
            rescaled_W2_sq = torch.sum(((rescaled_quantile_forecasts[:,:-1,:]-
                                         rescaled_quantile_forecasts_lagged[:,1:,:])**2)*
                                       (1/n_quantile_levels), -1)
            rescaled_wcW2_sq = torch.sum(((rescaled_quantile_forecasts[:,:-1,:]-
                                           rescaled_quantile_forecasts_lagged[:,1:,:])**2)*
                                         (quantile_levels*(1-quantile_levels))*
                                         (1/n_quantile_levels), -1)
            rescaled_wtW2_sq = torch.sum(((rescaled_quantile_forecasts[:,:-1,:]-
                                           rescaled_quantile_forecasts_lagged[:,1:,:])**2)*
                                         ((2*quantile_levels-1)**2)*
                                         (1/n_quantile_levels), -1)
            # Divide rescaled_W2_sq by scaling_constant before taking sqrt and mean across horizons and samples
            sW2 = torch.mean(torch.sqrt(rescaled_W2_sq/scaling_constant_sq.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1)))
            swcW2 = torch.mean(torch.sqrt(rescaled_wcW2_sq/scaling_constant_sq.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1)))
            swtW2 = torch.mean(torch.sqrt(rescaled_wtW2_sq/scaling_constant_sq.unsqueeze(-1).expand(-1, self.hparams.forecast_length-1)))

        # Calculate composite loss - 9 different combinations per loss_stability_type
        loss = 0
        w_quality = 0
        w_stability = 0
        if not calculate_metrics:
            # Step 1 = select loss components
            if loss_stability_type == 'W1': # loss_stability_type == 'W1'
                if wc_quality:
                    if wc_stability:
                        losses = torch.stack((loss_wc_quality, loss_wc_stability1))
                    elif wt_stability:
                        losses = torch.stack((loss_wc_quality, loss_wt_stability1))
                    else: 
                        losses = torch.stack((loss_wc_quality, loss_stability1))
                elif wt_quality:
                    if wc_stability:
                        losses = torch.stack((loss_wt_quality, loss_wc_stability1))
                    elif wt_stability:
                        losses = torch.stack((loss_wt_quality, loss_wt_stability1))
                    else: 
                        losses = torch.stack((loss_wt_quality, loss_stability1))
                else: 
                    if wc_stability:
                        losses = torch.stack((loss_quality, loss_wc_stability1))
                    elif wt_stability:
                        losses = torch.stack((loss_quality, loss_wt_stability1))
                    else: 
                        losses = torch.stack((loss_quality, loss_stability1))
            else: # loss_stability_type == 'W2'
                if wc_quality:
                    if wc_stability:
                        losses = torch.stack((loss_wc_quality, loss_wc_stability2))
                    elif wt_stability:
                        losses = torch.stack((loss_wc_quality, loss_wt_stability2))
                    else: 
                        losses = torch.stack((loss_wc_quality, loss_stability2))
                elif wt_quality:
                    if wc_stability:
                        losses = torch.stack((loss_wt_quality, loss_wc_stability2))
                    elif wt_stability:
                        losses = torch.stack((loss_wt_quality, loss_wt_stability2))
                    else: 
                        losses = torch.stack((loss_wt_quality, loss_stability2))
                else: 
                    if wc_stability:
                        losses = torch.stack((loss_quality, loss_wc_stability2))
                    elif wt_stability:
                        losses = torch.stack((loss_quality, loss_wt_stability2))
                    else: 
                        losses = torch.stack((loss_quality, loss_stability2))
            # Step 2 = combine loss components
            w_stability = self.hparams.lambda_stability
            w_quality = 1 - w_stability
            loss = w_quality * losses[0] + w_stability * losses[1]
                                        
        return (rescaled_quantile_forecasts, rescaled_quantile_forecasts_lagged, 
                loss_quality, sCRPS,
                loss_wc_quality, swcCRPS,
                loss_wt_quality, swtCRPS,
                logS,
                loss_stability1, sW1,
                loss_wc_stability1, swcW1,
                loss_wt_stability1, swtW1,
                loss_stability2, sW2,
                loss_wc_stability2, swcW2,
                loss_wt_stability2, swtW2,
                loss, w_quality, w_stability,
                bs)
                     
    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=self.hparams.weight_decay)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, amsgrad=False)
        scheduler = {
            # 'scheduler': torch.optim.lr_scheduler.SequentialLR(optimizer, 
            #                                                    schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10),
            #                                                                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.explr_gamma)],
            #                                                    milestones=[10]),
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.explr_gamma),
            'interval': 'epoch',  # Update the learning rate after each epoch
            'frequency': 1,       # Apply the scheduler every epoch
        }

        return [optimizer], [scheduler]  # Return the optimizer and scheduler as lists
        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        ensemble_id_update = batch_idx % self.hparams.ensemble_size
        for idx, model in enumerate(self.model):
            if idx != ensemble_id_update:
                for param in model.parameters():
                    param.grad = None
        optimizer.step(closure=optimizer_closure)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        #if self.global_step > 100: almost no impact for ema_decay = .99
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = self.hparams.ema_decay * ema_param.data + (1 - self.hparams.ema_decay) * model_param.data

    # # Hooks for using a rougher approximation of CRPS in first epochs/batches
    # ## Option 1
    # def on_train_start(self):
    #     self.target_n_quantile_levels_training = self.hparams.n_quantile_levels_training
    # def on_train_epoch_start(self):
    #     epoch = self.current_epoch
    #     if (epoch == 0): 
    #         self.hparams.n_quantile_levels_training = 5
    #     if (epoch > 0) and (epoch % 2 == 0) and (self.hparams.n_quantile_levels_training < self.target_n_quantile_levels_training):
    #         self.hparams.n_quantile_levels_training = min(
    #             self.hparams.n_quantile_levels_training*2,
    #             self.target_n_quantile_levels_training
    #         )
    #     print(f"Epoch {epoch}: n_q_train is {self.hparams.n_quantile_levels_training}")
    # ## Option 2
    # def on_train_start(self):
    #     self.target_n_quantile_levels_training = self.hparams.n_quantile_levels_training
    # def on_train_batch_start(self, batch, batch_idx):
    #     global_step = self.global_step
    #     if (global_step == 0):
    #         self.hparams.n_quantile_levels_training = 5
    #     if (global_step > 0) and (global_step % 200 == 0) and (self.hparams.n_quantile_levels_training < self.target_n_quantile_levels_training):
    #         self.hparams.n_quantile_levels_training = min(
    #             self.hparams.n_quantile_levels_training*2,
    #             self.target_n_quantile_levels_training
    #         )
    #     print(f"Step {global_step}: n_q_train is {self.hparams.n_quantile_levels_training}")
    
