# LOAD MODULES

# Standard library
import os

# Proprietary
from src.methods.StableSQF import LitNBEATSQF
from src.utils.callbacks import LoadModelWarning, WriteQuantileForecastsToCSV #, PlotTestPredictions, PlotValidationPredictions

# Third party
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
import torch

# CHECK CONNECTION W WANDB
wandb.login()
project_name = "StableSQF"

# EXPERIMENT CONFIGURATION

# Dataset
dataset = "M4"
subset = "Monthly"
dataset_id = "M4M"
validation_periods = 18
test_periods = 18
test_mode_nrows = None

# Model
load_model = False #True or False
update_loaded_model_specific_training_and_eval_hparams = False #True or False # only relevant when a model is loaded
swa = False #True or False

# Model architecture - load existing model or specify model hyperparameters
if load_model:
    model_id = "1wyjqom3"
    checkpoint = "last" #"best" or "last"
else: # hparams below are ignored if model is loaded
    backcast_length_multiplier = 8
    forecast_length = 6
    hidden_layer_units = 512 #256 #1024 #128 #64
    quantile_levels_knots = torch.tensor([0.01, 0.025, 0.05, 0.075, 0.1, #5
                                          0.1375, 0.175, 0.2125, 0.25, #4 
                                          0.2875, 0.325, 0.3625, 0.4, #4
                                          0.45, 0.5, 0.55, #3
                                          0.6, 0.6375, 0.675, 0.7125, #4                                          
                                          #0.75, 0.775, 0.8125, 0.85, #4
                                          0.75, 0.7875, 0.825, 0.8625, #4
                                          0.9, 0.925, 0.95, 0.975, 0.99]) #5
    spline_pieces = quantile_levels_knots.shape[0]+1
    wn = False #True
    n_blocks = 10
    n_blocks_shared = 1 #2 #0
    ensemble_size = 1 #2 #1
    zero_mean = True # model argument as it affects model weights
    unit_variance = True # model argument as it affects model weights

# Model training and evaluation
eval_mode = 'test' #'validation' or 'test'
random_seed = 1132 #2512 #20 #2512 #20 #18
## Data hparams
forecasting_origin_range_multiplier = 10 #1e6
batch_size = 512 #2048 #1024 #256 #128
## Model-specific training and evaluation hparams
if not load_model or update_loaded_model_specific_training_and_eval_hparams: 
# model-specific training and evaluation hparams below are ignored if (load_model == True) AND (update_loaded_model_specific_training_and_eval_hparams == False)
    n_quantile_levels_training = 100 # 100 = 0.005, 0.015 ... 0.9850, 0.9950
    n_quantile_levels_validation = 100 # 100 = 0.005, 0.015 ... 0.9850, 0.9950
    n_quantile_levels_test = 100 # 100 = 0.005, 0.015 ... 0.9850, 0.9950
    wc_quality = False # wCRPS center
    wt_quality = False # wCRPS tails
    loss_stability_type = 'W1' #'W2'] # Wasserstein distance
    wc_stability = False # wWx center
    wt_stability = False # wWx tails
    lambda_stability = 0.25 #0.1 #0.25 #0.20
    enforce_nonnegative_forecasts_metric_calculation = True
    optim_niter = 20 # nashmtl hyperparameter -- not implemented
    update_weights_every = 1 # nashmtl hyperparameter -- not implemented
    learning_rate = 1e-3 #4e-5 #1e-4
    explr_gamma = 1.0 #0.99 # decays lr as follows: learning_rate*(explr_gamma)**epoch
    weight_decay = 0.0 #1e-4
    ema_decay = 0.99
## Trainer hparams
max_norm = 1.0 # default
batches_per_epoch = 250 #50 #100
patience = 1e6 #20 #6 #10 #e6 #10 #20 #5 #10 #5 # 1e6 for specific number of epochs
max_epochs = 46 #50 #0 #200 #46 #0 # -1 for infinite training and 0 for zero-shot evaluation

# Other
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium") # if we run on GPU
save_forecasts = False

###################################################################################################
# Do not change anything below this line - only use for running experiment w config specified above
###################################################################################################

L.seed_everything(random_seed, workers=True)

# INIT MODEL
if load_model == True:
    path_to_checkpoint = "SQF_final/" + model_id + "/checkpoints/" + checkpoint + ".ckpt"
    NBEATSQF = LitNBEATSQF.load_from_checkpoint(path_to_checkpoint)
    # Extract model-dependent hyperparameters required for data loading
    forecast_length = NBEATSQF.hparams["forecast_length"]
    backcast_length_multiplier = NBEATSQF.hparams["backcast_length_multiplier"]
    zero_mean = NBEATSQF.hparams["zero_mean"]
    unit_variance = NBEATSQF.hparams["unit_variance"]
    n_quantile_levels_test = NBEATSQF.hparams["n_quantile_levels_test"]
    if update_loaded_model_specific_training_and_eval_hparams == True:
        # Update hyperparameters that do not affect the model architecture
        NBEATSQF.hparams["n_quantile_levels_training"]=n_quantile_levels_training
        NBEATSQF.hparams["n_quantile_levels_validation"]=n_quantile_levels_validation
        NBEATSQF.hparams["n_quantile_levels_test"]=n_quantile_levels_test
        NBEATSQF.hparams["wc_quality"]=wc_quality
        NBEATSQF.hparams["wt_quality"]=wt_quality
        NBEATSQF.hparams["loss_stability_type"]=loss_stability_type
        NBEATSQF.hparams["wc_stability"]=wc_stability
        NBEATSQF.hparams["wt_stability"]=wt_stability
        NBEATSQF.hparams["lambda_stability"]=lambda_stability
        NBEATSQF.hparams["enforce_nonnegative_forecasts_metric_calculation"]=enforce_nonnegative_forecasts_metric_calculation
        NBEATSQF.hparams["optim_niter"]=optim_niter
        NBEATSQF.hparams["update_weights_every"]=update_weights_every
        NBEATSQF.hparams["learning_rate"]=learning_rate
        NBEATSQF.hparams["explr_gamma"]=explr_gamma
        NBEATSQF.hparams["weight_decay"]=weight_decay
        NBEATSQF.hparams["ema_decay"]=ema_decay
else:
    NBEATSQF = LitNBEATSQF(# Model hypers
                           backcast_length_multiplier=backcast_length_multiplier,
                           forecast_length=forecast_length,
                           hidden_layer_units=hidden_layer_units,
                           quantile_levels_knots=quantile_levels_knots,
                           spline_pieces=spline_pieces,
                           wn=wn,
                           n_blocks=n_blocks,
                           n_blocks_shared=n_blocks_shared,
                           ensemble_size=ensemble_size,
                           zero_mean=zero_mean,
                           unit_variance=unit_variance,
                           # Optim hypers
                           n_quantile_levels_training=n_quantile_levels_training,
                           n_quantile_levels_validation=n_quantile_levels_validation,
                           n_quantile_levels_test=n_quantile_levels_test,
                           wc_quality=wc_quality,
                           wt_quality=wt_quality,
                           loss_stability_type=loss_stability_type,
                           wc_stability=wc_stability,
                           wt_stability=wt_stability,
                           lambda_stability=lambda_stability,
                           enforce_nonnegative_forecasts_metric_calculation=enforce_nonnegative_forecasts_metric_calculation,
                           optim_niter=optim_niter,
                           update_weights_every=update_weights_every,
                           learning_rate=learning_rate,
                           explr_gamma=explr_gamma,
                           weight_decay=weight_decay,
                           ema_decay=ema_decay)
# NBEATSQF = torch.compile(NBEATSQF)

# LOAD DATA(LOADERS)
if dataset == "M3":
    from src.data.M3 import load_data
if dataset == "M4":
    from src.data.M4 import load_data
if dataset == "M3_M3M4_train":
    from src.data.M3_M3M4_train import load_data
if dataset == "M3_M3M4_train_and_validation":
    from src.data.M3_M3M4_train_and_validation import load_data

train_dataloader, validation_dataloader, validation_dataloader_target, trainandvalidation_dataloader, test_dataloader_target = None, None, None, None, None
if eval_mode == "validation":
    train_dataloader, validation_dataloader, validation_dataloader_target, _, _ = load_data(
        subset=subset,
        test_mode_nrows=test_mode_nrows,
        backcast_length_multiplier=backcast_length_multiplier,
        forecast_length=forecast_length,
        validation_periods=validation_periods,
        test_periods=test_periods,
        zero_mean=zero_mean,
        unit_variance=unit_variance,
        forecasting_origin_range_multiplier=int(forecasting_origin_range_multiplier),
        batch_size=batch_size)
else:
    _, _, _, trainandvalidation_dataloader, test_dataloader_target = load_data(
        subset=subset,
        test_mode_nrows=test_mode_nrows,
        backcast_length_multiplier=backcast_length_multiplier,
        forecast_length=forecast_length,
        validation_periods=validation_periods,
        test_periods=test_periods,
        zero_mean=zero_mean,
        unit_variance=unit_variance,
        forecasting_origin_range_multiplier=int(forecasting_origin_range_multiplier),
        batch_size=batch_size)

# Check output dataloaders
# x_train, y_train = next(iter(train_dataloader))
# for batch_idx, (x_train, _) in enumerate(train_dataloader):
#     print("Batch:", batch_idx, "Input shape:", x_train["encoder_cont"].shape)
# # Check randomness over batches and epochs
# for batch_idx, (x_train, _) in enumerate(train_dataloader):
#     if batch_idx < 2 or batch_idx==91:
#         print("Batch:", batch_idx, "Input shape:", x_train["decoder_cont"][:,:,4].shape)
#         print("Batch:", batch_idx, "Input:", x_train["decoder_cont"][0,:,4])
#         print("Batch:", batch_idx, "Input sum:", x_train["decoder_cont"][:,:,4].sum(dim=0))
# for batch_idx, (x_train, _) in enumerate(train_dataloader):
#     if batch_idx < 2 or batch_idx==91:
#         print("Batch:", batch_idx, "Input shape:", x_train["decoder_cont"][:,:,4].shape)
#         print("Batch:", batch_idx, "Input:", x_train["decoder_cont"][0,:,4])
#         print("Batch:", batch_idx, "Input sum:", x_train["decoder_cont"][:,:,4].sum(dim=0))
# for batch_idx, (x_train, _) in enumerate(train_dataloader):
#     if batch_idx < 2 or batch_idx==91:
#         print("Batch:", batch_idx, "Input shape:", x_train["decoder_cont"][:,:,4].shape)
#         print("Batch:", batch_idx, "Input:", x_train["decoder_cont"][0,:,4])
#         print("Batch:", batch_idx, "Input sum:", x_train["decoder_cont"][:,:,4].sum(dim=0))
# for batch_idx, (x_validation, _) in enumerate(validation_dataloader):
#     print("Batch:", batch_idx, "Input shape:", x_validation["encoder_cont"].shape)

# CREATE TRAINER
# Create wandb logger and log config vars and hyperparameters that are not yet automatically logged to wandb
wandb_logger = WandbLogger(project=project_name, log_model=True) #, save_dir="/content/drive/My Drive/Colab Notebooks/SQF")
# Init callbacks that will be used in Trainer
modelsummary_callback = ModelSummary(max_depth=3)#-1)
# plot_test_predictions_callback = PlotTestPredictions()
if save_forecasts == True:
    write_quantile_forecasts = WriteQuantileForecastsToCSV(n_quantile_levels_test=n_quantile_levels_test, wandb_logger=wandb_logger)
if load_model == True:
    load_model_warning = LoadModelWarning(model_id=model_id)
if swa == True:
    swa_callback = StochasticWeightAveraging(swa_lrs=learning_rate, swa_epoch_start=0.0, annealing_epochs=0, annealing_strategy="linear", device=None)

# TRAIN AND EVALUATE MODEL
print("Start model training and evaluation.")
if eval_mode == 'validation':
    # Init additional callbacks that will be used in Trainer
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="vloss", mode="min", save_last=True)
    early_stop_callback = EarlyStopping(monitor="vloss", mode="min", patience=int(patience))
    callbacks_list = [
        modelsummary_callback,
        checkpoint_callback,
        early_stop_callback,
        # plot_test_predictions_callback,
    ]
    if save_forecasts == True:
        callbacks_list.append(write_quantile_forecasts)
    if load_model == True:
        callbacks_list.append(load_model_warning)
    if swa == True:
        callbacks_list.append(swa_callback)
    # Init Trainer
    trainer = Trainer(
        callbacks=callbacks_list,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=max_norm,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        #deterministic=False,
        max_epochs=max_epochs,
        #val_check_interval=100,
        limit_train_batches=batches_per_epoch,
        #log_every_n_steps=1,
    )
    # Fit
    trainer.fit(NBEATSQF, train_dataloader, validation_dataloader)
    # Evaluate
    if max_epochs == 0:
        trainer.test(NBEATSQF, validation_dataloader_target, verbose=True)
    else:
        trainer.test(NBEATSQF, validation_dataloader_target, "last", verbose=True) # test metrics are saved in ./wandb/run-.../files/output.log
        #trainer.test(NBEATSQF, validation_dataloader_target, "best", verbose=True) # test metrics are saved in ./wandb/run-.../files/output.log
elif eval_mode ==  'test':
    # Init additional callbacks that will be used in Trainer
    checkpoint_callback = ModelCheckpoint(save_last=True)
    callbacks_list = [
        modelsummary_callback,
        checkpoint_callback,
        # plot_test_predictions_callback,
    ]
    if save_forecasts == True:
        callbacks_list.append(write_quantile_forecasts)
    if load_model == True:
        callbacks_list.append(load_model_warning)
    if swa == True:
        callbacks_list.append(swa_callback)
    # Init Trainer
    trainer = Trainer(
        callbacks=callbacks_list,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=max_norm,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        #deterministic=False,
        max_epochs=max_epochs,
        #val_check_interval=100,
        limit_train_batches=batches_per_epoch,
    )
    # Fit
    trainer.fit(NBEATSQF, trainandvalidation_dataloader)
    # Evaluate
    if max_epochs == 0:
        trainer.test(NBEATSQF, test_dataloader_target, verbose=True)
    else:
        trainer.test(NBEATSQF, test_dataloader_target, "last", verbose=True) # test metrics are saved in ./wandb/run-.../files/output.log
        
# LOG HPARAM (UPDATES)
if load_model == True:
    wandb_logger.experiment.config.update({
        "dataset": dataset_id,
        "random_seed": random_seed,
        "origin_range": forecasting_origin_range_multiplier,
        "batch_size": batch_size,
        "patience": patience,
        "max_epochs": max_epochs,
        "max_norm": max_norm,
        "eval_mode": eval_mode,
        "n_batches": batches_per_epoch,
    })
    if update_loaded_model_specific_training_and_eval_hparams == True:
        wandb_logger.experiment.config.update({
            "n_quantile_levels_training": n_quantile_levels_training,
            "n_quantile_levels_validation": n_quantile_levels_validation,
            "n_quantile_levels_test": n_quantile_levels_test,
            "wc_quality": wc_quality,
            "wt_quality": wt_quality,
            "loss_stability_type": loss_stability_type,
            "wc_stability": wc_stability,
            "wt_stability": wt_stability,
            "lambda_stability": lambda_stability,
            "enforce_nonnegative_forecasts_metric_calculation": enforce_nonnegative_forecasts_metric_calculation,
            "optim_niter": optim_niter,
            "update_weights_every": update_weights_every,
            "learning_rate": learning_rate,
            "explr_gamma": explr_gamma,
            "weight_decay": weight_decay,
            "ema_decay": ema_decay,
        }, allow_val_change=True)
else:
    wandb_logger.experiment.config["dataset"] = dataset_id
    wandb_logger.experiment.config["random_seed"] = random_seed
    wandb_logger.experiment.config["origin_range"] = forecasting_origin_range_multiplier
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["patience"] = patience
    wandb_logger.experiment.config["max_epochs"] = max_epochs
    wandb_logger.experiment.config["max_norm"] = max_norm
    wandb_logger.experiment.config["eval_mode"] = eval_mode
    wandb_logger.experiment.config["n_batches"] = batches_per_epoch

wandb.finish()
