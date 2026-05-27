# LOAD MODULES

# Standard library
from typing import Tuple, Optional
import warnings

# Proprietary
from src.data.utils import (
    ts_train_val_test_split, 
    wide_to_long_univariate, 
    prepend_time_series_univariate, 
    scale_targets,
    DataFrame_to_TimeSeriesDataSet,
    TimeSeriesDataSet_to_DataLoader,
    summary_scaler,
)

# Third party
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

data_path = "./data/raw"
subset = "items"
test_mode_nrows = None
backcast_length_multiplier = int(26)
forecast_length = int(14)
validation_periods = int(28)
test_periods = int(28)
zero_mean = True
unit_variance = False
forecasting_origin_range_multiplier = int(2)
batch_size = int(32)

# DATA LOADING FUNCTION
def load_data(
    data_path: str = "./data/raw",
    subset: str = "items",
    test_mode_nrows: Optional[int] = None,
    backcast_length_multiplier: int = 26,
    forecast_length: int = 14,
    validation_periods: int = 28,
    test_periods: int = 28,
    zero_mean: bool = True,
    unit_variance: bool = False,
    forecasting_origin_range_multiplier: int = 2,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Loads and preprocesses M5 data.

    The function loads the data,
    optionally scales the data (based on mean target over train and validation periods),
    and preprocesses the data so that it can be transformed into PyTorch Forecasting TimeSeriesDataSets.
    A TimeSeriesDataSet can be easily transformed into a DataLoader.

    forecasting_origin_range_multiplier should be >= 0 and determines the size of the training set(s):
    = 0 --> (1+0*forecast_length) input-output window per time series;
    = 1 --> (1+1*forecast_length) input-output windows per time series.
    = 1e6 --> all possible input-output windows per time series.
    Low values of forecasting_origin_range_multiplier --> more recent input-output windows only.

    Returns: DataLoaders for the train, validation, trainandvalidation and test sets.
    'encoder_cont' contains batch_size x backcast_length x [static_reals+(time_varying_unknown_reals=encoder_target)].
    To retrieve input: x_train['encoder_cont'][..., -1]
    """
    # Some checks
    if (forecast_length<2):
        raise ValueError("A forecast_length of at least 2 is required for stability calculations")
    if (subset=="items") and (forecast_length!=14):
        warnings.warn("Are you sure you want to use a forecast_length value other than 14?")

    # Load/create TimeSeriesDataSet
    data_file_path = ("./data/processed/M5_" +
                      subset +
                      str(backcast_length_multiplier) +
                      str(forecast_length) +
                      str(validation_periods) +
                      str(test_periods) +
                      str(zero_mean) +
                      str(unit_variance) +
                      str(forecasting_origin_range_multiplier))

    if (test_mode_nrows==None) and (os.path.exists(data_file_path + "_train_set.pt")):
        
        print("TimeSeriesDataSets already exists!")

        train_set = torch.load(data_file_path + "_train_set.pt")
        validation_set = torch.load(data_file_path + "_validation_set.pt")
        validation_set_target = torch.load(data_file_path + "_validation_set_target.pt")
        trainandvalidation_set = torch.load(data_file_path + "_trainandvalidation_set.pt")
        test_set_target = torch.load(data_file_path + "_test_set_target.pt")             

    else:
        
        print("Creating TimeSeriesDataSets.")

        # Load raw data
        # dataset = pd.read_csv(f"/teamspace/studios/this_studio/{data_path}/m5_{subset}.txt", nrows=test_mode_nrows, sep=',', decimal='.', header=None, engine='python')
        dataset = pd.read_csv(f"{data_path}/m5_{subset}.txt", nrows=test_mode_nrows, sep=',', decimal='.', header=None, engine='python')
        # Get the data in np representation
        dataset = dataset.to_numpy()
        # Select all non NaN values
        dataset = [x[x == x] for x in dataset]
        # Remove leading zeros
        dataset = [np.trim_zeros(arr, 'f') for arr in dataset]
        df = wide_to_long_univariate(dataset) # required for transformation to TimeSeriesDataSet

        # Add 'lagged_value' column for instability calculations
        df['lagged_value'] = df.groupby('group')['value'].shift(periods=1, fill_value=None)

        # Prepend time series w none if number of observations is too low --> ensure that each time series is included in train_set
        min_num_observations = ((backcast_length_multiplier * forecast_length + forecast_length) + validation_periods + test_periods)
        df = prepend_time_series_univariate(df, min_num_observations)

        # Add scaling constants for summarizing results
        df = summary_scaler(df)
        
        # Train/val/test split
        train_set, validation_set, test_set = ts_train_val_test_split(df, validation_periods, test_periods)
        max_train_set_time_idx = train_set['time_idx'].max()
        max_validation_set_time_idx = validation_set['time_idx'].max()
        max_test_set_time_idx = test_set['time_idx'].max()

        # Zero-mean and/or unit-variance
        scaling_vars = validation_set.groupby('group')['value'].agg(['mean', 'std']).reset_index()
        if zero_mean is False: 
            scaling_vars['mean'] = 0
        if unit_variance is False:
            scaling_vars['std'] = 1
        train_set, validation_set, test_set = scale_targets(train_set, validation_set, test_set, scaling_vars)

        # Replace NaNs
        train_set.fillna(0, inplace=True)
        validation_set.fillna(0, inplace=True)
        test_set.fillna(0, inplace=True)
        
        # Create the TimeSeriesDataSets from the pandas DataFrames - 'time_idx' is right-aligned (see ts_train_val_test_split)
        train_set, validation_set, validation_set_target, trainandvalidation_set, test_set_target = DataFrame_to_TimeSeriesDataSet(
            train_set, 
            validation_set,
            validation_set, # validation_set_target
            validation_set, # trainandvalidation_set
            test_set, # test_set_target
            backcast_length_multiplier,
            forecast_length,
            forecasting_origin_range_multiplier,
            validation_periods,
            test_periods,
            max_train_set_time_idx,
            max_validation_set_time_idx,
            max_validation_set_time_idx, # max_validation_set_target_time_idx
            max_validation_set_time_idx, # max_trainandvalidation_set_time_idx
            max_test_set_time_idx) # max_test_set_target_time_idx

        # Save TimeSeriesDataSets
        torch.save(train_set, data_file_path + "_train_set.pt")
        torch.save(validation_set, data_file_path + "_validation_set.pt")
        torch.save(validation_set_target, data_file_path + "_validation_set_target.pt")
        torch.save(trainandvalidation_set, data_file_path + "_trainandvalidation_set.pt")
        torch.save(test_set_target, data_file_path + "_test_set_target.pt")

    # Create dataloaders
    train_dataloader, validation_dataloader, validation_dataloader_target, trainandvalidation_dataloader, test_dataloader_target = TimeSeriesDataSet_to_DataLoader(
        train_set,
        validation_set,
        validation_set_target, 
        trainandvalidation_set,
        test_set_target,
        batch_size
    )
    
    return train_dataloader, validation_dataloader,  validation_dataloader_target, trainandvalidation_dataloader, test_dataloader_target

# # Test sampling behavior
# bincount_list = []
# train_dataloader, validation_dataloader, validation_dataloader_target, trainandvalidation_dataloader, test_dataloader_target = load_data(
#         test_mode_nrows=100,
#         batch_size=32,
#         forecasting_origin_range_multiplier=1) # 2000
# for loop_id in range(1000):
#     x_train, y_train = next(iter(train_dataloader))
#     bincount_list.append(x_train['groups'].squeeze().bincount(minlength=100).numpy())
# stacked_bincounts = np.stack(bincount_list,0)
# np.mean(stacked_bincounts, 0)/32 # should all equal 1/100 = 1/test_mode_nrows

# # Test output load_data
# train_dataloader, validation_dataloader, validation_dataloader_target, trainandvalidation_dataloader, test_dataloader_target = load_data(
#     test_mode_nrows=2,
#     batch_size=2,
#     forecasting_origin_range_multiplier=0)
# x_train, y_train = next(iter(train_dataloader))
# print("x_train =", x_train)
# print("\ny_train =", y_train)
# print("\nsizes of x_train =")
# for key, value in x_train.items():
#     if isinstance(value, torch.Tensor):
#         print(f"\t{key} = {value.size()}")
#     else:
#         print(f"\t{key} = {len(value)}")
# lookback_window = x_train["encoder_cont"][:,:,4]
# lookback_window_lagged = x_train["encoder_cont"][:,:,5]
# # lookback_window2 = x_train["encoder_target"][0]
# # lookback_window_lagged2 = x_train["encoder_target"][1]
# forecast_period = x_train['decoder_cont'][:,:,4]
# forecast_period_lagged = x_train['decoder_cont'][:,:,5]
# #forecast_period2 = x_train['decoder_target'][0]
# #forecast_period_lagged2 = x_train['decoder_target'][1]
# #forecast_period3 = y_train[0][0]
# #forecast_period_lagged3 = y_train[0][1]
# mean = x_train["decoder_cont"][:,:,0]
# std = x_train["decoder_cont"][:,:,1]
# # average one-step naive forecast computed in-sample 
# # based on lookback window AND (non-zero) values before lookback window
# scaling_constant_abs = x_train["encoder_cont"][:,-1,2]
# scaling_constant_sq = x_train["encoder_cont"][:,-1,3]
