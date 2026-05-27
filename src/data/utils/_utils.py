# LOAD MODULES

# Standard library
from typing import Tuple
import math
import os # to set num_workers in DataLoaders > 0 - seems to be incompatible with other config settings (due to batch_sampling?)

# Proprietary

# Third party
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    MultiNormalizer,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

# FUNCTIONS
def ts_train_val_test_split(
    data: pd.DataFrame,
    validation_periods: int,
    test_periods: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a pd.DataFrame that contains all data into a training, validation, and test set.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame (long format) with a column 'group' that serves as time series identifier,
    and a column 'time_idx' that serves at time period identifier.
    validation_periods (int): The number of periods that are only used for validation.
    test_periods (int): The number of periods that are only used for testing.

    Returns:
    tuple: A tuple containing three pandas DataFrames. 
    The first DataFrame contains the training set, the second contains the validation set, and the third contains the test set.
    """
    # Split the DataFrame into train, validation, and test sets
    test_set = data.copy()  # all data points
    validation_set = data.groupby('group').apply(lambda x: x.iloc[:-test_periods]) # drop last test_periods of each time series
    train_set = data.groupby('group').apply(lambda x: x.iloc[:-(validation_periods+test_periods)]) # drop validation and test periods of each time series

    # Reset index for all sets
    test_set.reset_index(drop=True, inplace=True)
    validation_set.reset_index(drop=True, inplace=True)
    train_set.reset_index(drop=True, inplace=True)

    # Transform 'time_idx' column to be right-aligned within each group
    max_test_set_time_idx = test_set['time_idx'].max()
    max_test_set_group_time_idx = test_set.groupby('group')['time_idx'].transform('max')
    test_set['time_idx'] = test_set['time_idx'] + (max_test_set_time_idx - max_test_set_group_time_idx)
    max_validation_set_time_idx = validation_set['time_idx'].max()
    max_validation_set_group_time_idx = validation_set.groupby('group')['time_idx'].transform('max')
    validation_set['time_idx'] = validation_set['time_idx'] + (max_validation_set_time_idx - max_validation_set_group_time_idx)
    max_train_set_time_idx = train_set['time_idx'].max()
    max_train_set_group_time_idx = train_set.groupby('group')['time_idx'].transform('max')
    train_set['time_idx'] = train_set['time_idx'] + (max_train_set_time_idx - max_train_set_group_time_idx)

    return train_set, validation_set, test_set

def wide_to_long_univariate(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    From wide to long (flatten data) - required for transformation to TimeSeriesDataSet.
    """
    # Initialize lists to store flattened data
    values = []
    groups = []
    time_indices = []

    # Iterate over the list of np arrays
    for group, ts in enumerate(data):
        values.extend(ts)
        groups.extend([group] * len(ts))
        time_indices.extend(range(len(ts)))
    
    # Create a pandas DataFrame from the flattened data
    df = pd.DataFrame({'value': values, 'group': groups, 'time_idx': time_indices})

    return df

def prepend_time_series_univariate(
    df: pd.DataFrame, 
    min_num_observations: int
) -> pd.DataFrame:
    """
    Prepend additional rows (add zero observations) to time series if number of observations is too low.
    Input df should be in long format.
    """
    group_num_observations = df.groupby('group').size()

    rows_to_prepend = []
    # For each group with less than 'min_num_observations' observations, prepend additional rows with 'value' and 'lagged_value' 0
    for group_id, count in group_num_observations[group_num_observations < min_num_observations].items():
        obs_to_prepend = min_num_observations - count
        min_time_idx = df.loc[df['group'] == group_id, 'time_idx'].min()
        df.loc[df['group'] == group_id, 'time_idx'] += obs_to_prepend
        rows_to_prepend.extend([{
            'value': None, 
            'group': group_id, 
            'time_idx': time_idx,
            'lagged_value': None
        } for time_idx in range(min_time_idx, obs_to_prepend)])
    if rows_to_prepend:
        df = pd.concat([pd.DataFrame(rows_to_prepend), df], ignore_index=True)
    df = df.sort_values(by=['group', 'time_idx']).reset_index(drop=True)
    # print(df.groupby('group').count())
    
    return df

def summary_scaler(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add scaling constants for summarizing results.
    """
    df = df.sort_values(by=['group', 'time_idx'])
    df['diff'] =  df.groupby('group')['value'].diff()
    df['diff_abs'] = df['diff'].abs()
    df['diff_sq'] = df['diff']**2

    df['scaling_constant_abs'] = df.groupby('group')['diff_abs'].expanding().mean().reset_index(level=0, drop=True)
    df['scaling_constant_sq'] = df.groupby('group')['diff_sq'].expanding().mean().reset_index(level=0, drop=True)
    df['scaling_constant_abs'] = df['scaling_constant_abs'] + 1e-3 # for numerical stability
    df['scaling_constant_sq'] = df['scaling_constant_sq'] + 1e-3 # for numerical stability
    
    return df

def scale_targets(
    train_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    test_set: pd.DataFrame,
    scaling_vars: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scale targets and lagged targets.
    """
    train_set = train_set.merge(scaling_vars, on='group', how='left')
    train_set['value'] = (train_set['value'] - train_set['mean']) / train_set['std']
    train_set['lagged_value'] = (train_set['lagged_value'] - train_set['mean']) / train_set['std']
    validation_set = validation_set.merge(scaling_vars, on='group', how='left')
    validation_set['value'] = (validation_set['value'] - validation_set['mean']) / validation_set['std']
    validation_set['lagged_value'] = (validation_set['lagged_value'] - validation_set['mean']) / validation_set['std']
    test_set = test_set.merge(scaling_vars, on='group', how='left')
    test_set['value'] = (test_set['value'] - test_set['mean']) / test_set['std']
    test_set['lagged_value'] = (test_set['lagged_value'] - test_set['mean']) / test_set['std']

    return train_set, validation_set, test_set

def DataFrame_to_TimeSeriesDataSet(
    train_set: pd.DataFrame, 
    validation_set: pd.DataFrame, 
    validation_set_target: pd.DataFrame,
    trainandvalidation_set: pd.DataFrame,
    test_set_target: pd.DataFrame,
    backcast_length_multiplier: int,
    forecast_length: int,
    forecasting_origin_range_multiplier: int,
    validation_periods: int,
    test_periods: int,
    max_train_set_time_idx: int,
    max_validation_set_time_idx: int,
    max_validation_set_target_time_idx: int,
    max_trainandvalidation_set_time_idx: int,
    max_test_set_target_time_idx: int
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Create TimeSeriesDataSets from the pandas DataFrames - 'time_idx' is right-aligned.
    """
    # Scaling - GroupNormalizer and EncoderNormalizer both perform standard scaling = standardization
    # Scaling - GroupNormalizer(groups=["group"]) = standardization per group = per time series
    # Scaling - EncoderNormalizer(method='standard') = standardization per sample
    # Scaling - EncoderNormalizer(method='identity') = use raw data
    # Scaling - However, as we do not generate predictions for the target directly, but for the parameters of the spline_qf
    # Scaling - it is not possible to directly use this.
    # Scaling - Instead, we can use zero_mean and unit_variance (per time series)
    # Scaling - and pass the scaling variables to static_reals for later use.
    train_set = TimeSeriesDataSet(
        train_set, group_ids=["group"], target=["value", "lagged_value"], time_idx="time_idx",
        max_encoder_length=backcast_length_multiplier*forecast_length,
        max_prediction_length=forecast_length,
        # Scaling
        time_varying_unknown_reals=["value", "lagged_value"], # if provided, it fills en/decoder_cont with scaled en/decoder_target
        static_reals=["mean", "std", "scaling_constant_abs", "scaling_constant_sq"],
        target_normalizer=MultiNormalizer([EncoderNormalizer(method="identity"),
                                           EncoderNormalizer(method="identity"),
                                           ]),
        scalers={"mean": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "std": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_abs": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_sq": StandardScaler(copy=False, with_mean=False, with_std=False),
                 }, # for scaling covariates - here = to prevent scaling 'mean', 'std', and 'scaling_constant'
        # Train/val/test characteristics
        predict_mode=False,
        min_prediction_idx=math.floor(
            (max_train_set_time_idx-forecast_length+1) # only one possible input-output window
            - (forecasting_origin_range_multiplier*forecast_length) # allow for (multiplier*forecast_length) additional input-output windows
            ), # 'time_idx' is right-aligned
    )
    # print(len(train_set)) # = equal to number of time series 
    validation_set = TimeSeriesDataSet(
        validation_set, group_ids=["group"], target=["value", "lagged_value"], time_idx="time_idx",
        max_encoder_length=backcast_length_multiplier*forecast_length,
        max_prediction_length=forecast_length,
        # Scaling
        time_varying_unknown_reals=["value", "lagged_value"], # if provided, it fills en/decoder_cont with scaled en/decoder_target
        static_reals=["mean", "std", "scaling_constant_abs", "scaling_constant_sq"],
        target_normalizer=MultiNormalizer([EncoderNormalizer(method="identity"),
                                           EncoderNormalizer(method="identity"),
                                           ]),
        scalers={"mean": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "std": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_abs": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_sq": StandardScaler(copy=False, with_mean=False, with_std=False),
                 }, # for scaling covariates - here = to prevent scaling 'mean', 'std', and 'scaling_constant'
        # Train/val/test characteristics
        predict_mode=False, # (=True) does not allow for rolling origin evaluation (but only creates one sample per time series)
        min_prediction_idx=(max_validation_set_time_idx-validation_periods+1), # 'time_idx' is right-aligned
    )
    validation_set_target = TimeSeriesDataSet(
        validation_set_target, group_ids=["group"], target=["value", "lagged_value"], time_idx="time_idx",
        max_encoder_length=backcast_length_multiplier*forecast_length,
        max_prediction_length=forecast_length,
        # Scaling
        time_varying_unknown_reals=["value", "lagged_value"], # if provided, it fills en/decoder_cont with scaled en/decoder_target
        static_reals=["mean", "std", "scaling_constant_abs", "scaling_constant_sq"],
        target_normalizer=MultiNormalizer([EncoderNormalizer(method="identity"),
                                           EncoderNormalizer(method="identity"),
                                           ]),
        scalers={"mean": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "std": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_abs": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_sq": StandardScaler(copy=False, with_mean=False, with_std=False),
                 }, # for scaling covariates - here = to prevent scaling 'mean', 'std', and 'scaling_constant'
        # Train/val/test characteristics
        predict_mode=False, # (=True) does not allow for rolling origin evaluation (but only creates one sample per time series)
        min_prediction_idx=(max_validation_set_target_time_idx-validation_periods+1), # 'time_idx' is right-aligned
    )
    trainandvalidation_set = TimeSeriesDataSet(
        trainandvalidation_set, group_ids=["group"], target=["value", "lagged_value"], time_idx="time_idx",
        max_encoder_length=backcast_length_multiplier*forecast_length,
        max_prediction_length=forecast_length,
        # Scaling
        time_varying_unknown_reals=["value", "lagged_value"], # if provided, it fills en/decoder_cont with scaled en/decoder_target
        static_reals=["mean", "std", "scaling_constant_abs", "scaling_constant_sq"],
        target_normalizer=MultiNormalizer([EncoderNormalizer(method="identity"),
                                           EncoderNormalizer(method="identity"),
                                           ]),
        scalers={"mean": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "std": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_abs": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_sq": StandardScaler(copy=False, with_mean=False, with_std=False),
                 }, # for scaling covariates - here = to prevent scaling 'mean', 'std', and 'scaling_constant'
        # Train/val/test characteristics
        predict_mode=False, # (=True) does not allow for rolling origin evaluation (but only creates one sample per time series)
        min_prediction_idx=math.floor(
            (max_trainandvalidation_set_time_idx-forecast_length+1) # only one possible input-output window
            - (forecasting_origin_range_multiplier*forecast_length) # allow for (multiplier*forecast_length) additional input-output windows
            ), # 'time_idx' is right-aligned
        #min_prediction_idx=(max_trainandvalidation_set_time_idx-validation_periods+1), # 'time_idx' is right-aligned
    )
    test_set_target = TimeSeriesDataSet(
        test_set_target, group_ids=["group"], target=["value", "lagged_value"], time_idx="time_idx",
        max_encoder_length=backcast_length_multiplier*forecast_length,
        max_prediction_length=forecast_length,
        # Scaling
        time_varying_unknown_reals=["value", "lagged_value"], # if provided, it fills en/decoder_cont with scaled en/decoder_target
        static_reals=["mean", "std", "scaling_constant_abs", "scaling_constant_sq"],
        target_normalizer=MultiNormalizer([EncoderNormalizer(method="identity"),
                                           EncoderNormalizer(method="identity"),
                                           ]),
        scalers={"mean": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "std": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_abs": StandardScaler(copy=False, with_mean=False, with_std=False),
                 "scaling_constant_sq": StandardScaler(copy=False, with_mean=False, with_std=False),
                 }, # for scaling covariates - here = to prevent scaling 'mean', 'std', and 'scaling_constant'
        # Train/val/test characteristics
        predict_mode=False, # (=True) does not allow for rolling origin evaluation (but only creates one sample per time series)
        min_prediction_idx=(max_test_set_target_time_idx-test_periods+1), # 'time_idx' is right-aligned
    )
    # print(train_set.index.groupby('sequence_id').size())
    # print(validation_set.index.groupby('sequence_id').size())
    # print(validation_set_target.index.groupby('sequence_id').size())
    # print(trainandvalidation_set.index.groupby('sequence_id').size())
    # print(test_set_target.index.groupby('sequence_id').size())

    return train_set, validation_set, validation_set_target, trainandvalidation_set, test_set_target

def TimeSeriesDataSet_to_DataLoader(
    train_set: TimeSeriesDataSet,
    validation_set: TimeSeriesDataSet,
    validation_set_target: TimeSeriesDataSet,
    trainandvalidation_set: TimeSeriesDataSet,
    test_set_target: TimeSeriesDataSet,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from TimeSeriesDataSets.
    """
    if torch.cuda.is_available(): # if we run on GPU
        batch_size_eval = 4096        
    else:
        batch_size_eval = 512
    # Sample weighting - TimeSeriesDataSet.index includes all possible input-output samples per group.
    # Sample weighting - If we use standard random sampling, long series are thus selected way more often.
    # Sample weighting - For training, we want each time series to have an equal chance of being selected (and a window is sampled only in a second step).
    # Sample weighting - Hence, we assign a sampling weight equal to (1/number of possible input-output samples).
    # Sample weighting - Number of possible input-output samples for specific group 
    # Sample weighting - = length of ts - (backcast_length_multiplier * forecast_length + forecast_length) + 1.
    sampling_weights = 1/(train_set.index['count']-train_set.index['sequence_length']+1)
    sampling_weights = sampling_weights.to_numpy()
    # print(sampling_weights)
    batch_sampling = WeightedRandomSampler(sampling_weights, len(sampling_weights))
    train_dataloader = train_set.to_dataloader(train=True, batch_size=batch_size,
                                               sampler=batch_sampling, shuffle=False,
                                               #timeout=120,
                                               num_workers=os.cpu_count()-1)
    # If forecasting_origin_range_multiplier is low enough (and with certainty if it is 0), we don't need the above
    # batch_sampling sampler to ensure that each time series is equally represented in the training batches, and we can use:
    # train_dataloader = train_set.to_dataloader(train=True, batch_size=batch_size)
    validation_dataloader = validation_set.to_dataloader(train=False, batch_size=batch_size_eval,
                                                         #timeout=120,
                                                         num_workers=os.cpu_count()-1)
    validation_dataloader_target = validation_set_target.to_dataloader(train=False, batch_size=batch_size_eval,
                                                                       #timeout=120,
                                                                       num_workers=os.cpu_count()-1)
    sampling_weights = 1/(trainandvalidation_set.index['count']-trainandvalidation_set.index['sequence_length']+1)
    sampling_weights = sampling_weights.to_numpy()
    # print(sampling_weights)
    batch_sampling = WeightedRandomSampler(sampling_weights, len(sampling_weights))
    trainandvalidation_dataloader = trainandvalidation_set.to_dataloader(train=True, batch_size=batch_size,
                                                                         sampler=batch_sampling, shuffle=False,
                                                                         #timeout=120,
                                                                         num_workers=os.cpu_count()-1)
    test_dataloader_target = test_set_target.to_dataloader(train=False, batch_size=batch_size_eval,
                                                           #timeout=120,
                                                           num_workers=os.cpu_count()-1)
    # For small datasets, it is possible to only use one batch:
    # validation_dataloader = validation_set.to_dataloader(train=False, batch_size=(n_groups*(validation_periods-forecast_length+1)))    
    # validation_dataloader_target = validation_set_target.to_dataloader(train=False, batch_size=(n_groups*(validation_periods-forecast_length+1)))    
    # test_dataloader_target = test_set_target.to_dataloader(train=False, batch_size=(n_groups*(test_periods-forecast_length+1)))
    # However, this quickly results in very slow validation/test processing if forecast_length < validation_periods/test_periods,
    # even for small datasets in terms of number of time series. If we also use a batch_size for validation and testing, 
    # the metrics are aggregated across the batches via 'mean' (which represents the mean value at sample level).
    # This ensures that training loss and validation/test loss are on the same scale.

    return train_dataloader, validation_dataloader, validation_dataloader_target, trainandvalidation_dataloader, test_dataloader_target
    