# LOAD MODULES

# Standard library
import csv
import os

from lightning import LightningModule, Trainer

# Proprietary
from src.utils.plotting import plot_quantile_forecasts

# Third party
import torch
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import Callback
import wandb

class PlotValidationPredictions(Callback):

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log n sample plots
        n = 1 # set to >4 to check M3M validation/test setup - last batch only contains 4 samples with a batch_size=32
        x_validation, _ = batch

        if len(x_validation["encoder_cont"].shape) == 2:
            x_validation["encoder_cont"] = x_validation["encoder_cont"].unsqueeze(0)
        if len(x_validation["decoder_cont"].shape) == 2:
            x_validation["decoder_cont"] = x_validation["decoder_cont"].unsqueeze(0)

        rescaled_lookback_window = x_validation["encoder_cont"][:,:,4] * x_validation["encoder_cont"][:,:,1] + x_validation["encoder_cont"][:,:,0]
        rescaled_quantile_forecasts = outputs
        rescaled_forecast_period = x_validation["decoder_cont"][:,:,4] * x_validation["decoder_cont"][:,:,1] + x_validation["decoder_cont"][:,:,0]

        rescaled_lookback_window = rescaled_lookback_window.cpu()
        rescaled_quantile_forecasts = rescaled_quantile_forecasts.cpu()
        rescaled_forecast_period = rescaled_forecast_period.cpu()

        available_samples = np.linspace(0, rescaled_quantile_forecasts.shape[0]-1, rescaled_quantile_forecasts.shape[0])
        selected_samples = np.random.choice(available_samples, size=n, replace=False)

        # plots = []
        # example_counter = trainer.current_epoch*n-1
        for i in selected_samples.tolist():
            # print(i)
            # example_counter += 1
            plot = plot_quantile_forecasts(rescaled_lookback_window, rescaled_quantile_forecasts, rescaled_forecast_period, int(i), trainer.current_epoch)
            # plots.append(plot)
            # wandb.log({"plots": plot})
            trainer.logger.experiment.log({"Examples": plot}) #step=example_counter

class PlotTestPredictions(Callback):

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # `outputs` comes from `LightningModule.test_step`
        # which corresponds to our model predictions in this case

        # Let's log n sample plots
        n = 1 # set to >4 to check M3M validation/test setup - last batch only contains 4 samples with a batch_size=32
        x_test, _ = batch

        if len(x_test["encoder_cont"].shape) == 2:
            x_test["encoder_cont"] = x_test["encoder_cont"].unsqueeze(0)
        if len(x_test["decoder_cont"].shape) == 2:
            x_test["decoder_cont"] = x_test["decoder_cont"].unsqueeze(0)

        rescaled_lookback_window = x_test["encoder_cont"][:,:,4] * x_test["encoder_cont"][:,:,1] + x_test["encoder_cont"][:,:,0]
        rescaled_quantile_forecasts = outputs[..., 0]
        rescaled_forecast_period = x_test['decoder_cont'][:,:,4] * x_test["decoder_cont"][:,:,1] + x_test["decoder_cont"][:,:,0]

        rescaled_lookback_window = rescaled_lookback_window.cpu()
        rescaled_quantile_forecasts = rescaled_quantile_forecasts.cpu()
        rescaled_forecast_period = rescaled_forecast_period.cpu()

        available_samples = np.linspace(0, rescaled_quantile_forecasts.shape[0]-1, rescaled_quantile_forecasts.shape[0])
        selected_samples = np.random.choice(available_samples, size=n, replace=False)

        # plots = []
        # example_counter = trainer.current_epoch*n-1
        for i in selected_samples.tolist():
            # print(i)
            # example_counter += 1
            plot = plot_quantile_forecasts(rescaled_lookback_window, rescaled_quantile_forecasts, rescaled_forecast_period, int(i), None)
            # plots.append(plot)
            # wandb.log({"plots": plot})
            trainer.logger.experiment.log({"Examples": plot}) #step=example_counter

class WriteQuantileForecastsToCSV(Callback):
    def __init__(self, n_quantile_levels_test, wandb_logger, filename1='quantile_forecasts.csv', filename2='quantile_forecasts_lagged.csv'):
        self.filepath = wandb_logger.experiment.dir
        self.filename1 = os.path.join(self.filepath, filename1)
        self.filename2 = os.path.join(self.filepath, filename2)

        # Ensure the directory exists
        os.makedirs(self.filepath, exist_ok=True)

        # Ensure the files are created and headers are written
        if not os.path.isfile(self.filename1):
            with open(self.filename1, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the headers
                headers = ['horizon', 'actual', 'scaling_constant_abs', 'scaling_constant_sq']
                headers += [f'q{i+1}' for i in range(n_quantile_levels_test)]
                writer.writerow(headers)
        if not os.path.isfile(self.filename2):
            with open(self.filename2, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the headers
                headers = ['horizon', 'actual', 'scaling_constant_abs', 'scaling_constant_sq']
                headers += [f'q{i+1}' for i in range(n_quantile_levels_test)]
                writer.writerow(headers)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # `outputs` comes from `LightningModule.test_step`
        # which corresponds to our model predictions in this case
        x_test, _ = batch

        # To handle batch_size = 1
        if len(x_test["decoder_cont"].shape) == 2:
            x_test["decoder_cont"] = x_test["decoder_cont"].unsqueeze(0)
        if len(x_test["encoder_cont"].shape) == 2:
            x_test["encoder_cont"] = x_test["encoder_cont"].unsqueeze(0)

        rescaled_forecast_period = x_test['decoder_cont'][:,:,4] * x_test["decoder_cont"][:,:,1] + x_test["decoder_cont"][:,:,0]
        rescaled_forecast_period_lagged = x_test['decoder_cont'][:,:,5] * x_test["decoder_cont"][:,:,1] + x_test["decoder_cont"][:,:,0]
        scaling_constant_abs = x_test["encoder_cont"][:,-1,2].unsqueeze(-1).expand(-1, x_test['decoder_cont'].shape[1])
        scaling_constant_sq = x_test["encoder_cont"][:,-1,3].unsqueeze(-1).expand(-1, x_test['decoder_cont'].shape[1])
        rescaled_quantile_forecasts = outputs[..., 0]
        rescaled_quantile_forecasts_lagged = outputs[..., 1]

        rescaled_forecast_period = rescaled_forecast_period.cpu().numpy()
        rescaled_forecast_period_lagged = rescaled_forecast_period_lagged.cpu().numpy()
        scaling_constant_abs = scaling_constant_abs.cpu().numpy()
        scaling_constant_sq = scaling_constant_sq.cpu().numpy()
        rescaled_quantile_forecasts = rescaled_quantile_forecasts.cpu().numpy()
        rescaled_quantile_forecasts_lagged = rescaled_quantile_forecasts_lagged.cpu().numpy()

        bs, fl, nql = rescaled_quantile_forecasts.shape
        horizons = np.tile(np.arange(fl), bs)
        
        rescaled_forecast_period_flat = rescaled_forecast_period.flatten()
        rescaled_forecast_period_lagged_flat = rescaled_forecast_period_lagged.flatten()
        scaling_constant_abs_flat = scaling_constant_abs.flatten()
        scaling_constant_sq_flat = scaling_constant_sq.flatten() 
        rescaled_quantile_forecasts_flat = rescaled_quantile_forecasts.reshape(bs * fl, nql)
        rescaled_quantile_forecasts_lagged_flat = rescaled_quantile_forecasts_lagged.reshape(bs * fl, nql)
        
        data1 = {
            'horizon': horizons,
            'actual': rescaled_forecast_period_flat,
            'scaling_constant_abs': scaling_constant_abs_flat,
            'scaling_constant_sq': scaling_constant_sq_flat
        }
        data2 = {
            'horizon': horizons,
            'actual': rescaled_forecast_period_lagged_flat,
            'scaling_constant_abs': scaling_constant_abs_flat,
            'scaling_constant_sq': scaling_constant_sq_flat
        }
        for i in range(nql):
            data1[f'q{i+1}'] = rescaled_quantile_forecasts_flat[:, i]
            data2[f'q{i+1}'] = rescaled_quantile_forecasts_lagged_flat[:, i]
        
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        # Write batch results to CSV
        with open(self.filename1, mode='a', newline='') as file:
            df1.to_csv(file, header=False, index=False)
        with open(self.filename2, mode='a', newline='') as file:
            df2.to_csv(file, header=False, index=False)

class LoadModelWarning(Callback):
    def __init__(self, model_id):
        self.model_id = model_id

    def on_train_start(self, trainer, pl_module):
        print("Model " + self.model_id + " is loaded.")
