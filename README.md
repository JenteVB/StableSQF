# *Stabilizing distribution-free probabilistic forecasts* </br><sub><sub>*Jente Van Belle, Honglin Wen, Wouter Verbeke, and Pierre Pinson* [[*Preprint 2026*]](https://arxiv.org/abs/2605.28531)</sub></sub>

*Multi-step-ahead forecasts are often updated as new observations become available, since shorter forecast horizons typically improve forecast quality. However, such improvements come at the cost of forecast instability, i.e., variability in forecasts for the same target period. This instability can trigger costly changes to plans formulated based on the forecasts and may erode trust in the forecasting system. In this work, we integrate forecast stability alongside forecast quality into the training of distribution-free probabilistic time-series forecasting models, allowing us to control this trade-off. We propose a method for generating stabilized forecasted conditional quantile functions using regression splines parameterized by a neural network. This approach enables joint optimization of stability and quality, as it allows us to directly penalize dissimilarities arising from forecast updates. Furthermore, it allows assigning varying importance to stabilizing different parts of the forecast distributions (e.g., central parts vs. tails) to focus on the parts most relevant for the intended downstream use (e.g., the upper tail for inventory management). We empirically evaluate the proposed method on two datasets with different statistical properties and show that it can effectively reduce forecast instability without a substantial loss in forecast quality, and that it can target stabilization effort toward specific parts of the forecast distributions.*

## Repository structure
This repository is organised as follows:
```bash
|- R scripts/
    |- Baselines.R           # ETS-G/B, mean-G/B, snaive-G/B + dataset summary statistics
    |- SQF-stabilized.R      # SQF-stabilized Partial and Full
|- data/
    |- processed/            # Saved TimeSeriesDataSets - for running experiments with same data structure (bl_multiplier, fc_length, val/test periods, input scaling, fo_range_multiplier)
    |- raw/
        |- m5_items.txt
|- src/
    |- data/
        |- M4.py
        |- M5.py
        |- utils/
    |- methods/
        |- StableSQF.py      # SQF forecaster model architecture and optimization procedure to stabilize the forecasts (StableSQF)
    |- utils/
|- main_M4_monthly.py        # Script to train a (Stable)SQF model on the M4 monthly dataset
|- main_M5_items.py          # Script to train a (Stable)SQF model on the M5 items dataset
```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Use
To efficiently train (Stable)SQF models, access to a CUDA-enabled GPU is required.

Weights & Biases is used for performance metric logging. Change the `project_name` in `main_*.py` to your `wandb` project in the script you want to run. A run produces `.csv` files with the forecasts if you set `save_forecasts = True` in `main_*.py` (see `WriteQuantileForecastsToCSV` callback in `src/utils/callbacks.py`). A `wandb` output folder is automatically created.

The `SQF-stabilized.R` script takes two of these generated `.csv` files as input: `quantile_forecasts.csv` and `quantile_forecasts_lagged.csv`.

## Citing
Please cite our paper and/or code as follows:
```tex
@article{vanbelle2026StableSQF,
  title     = {Stabilizing distribution-free probabilistic forecasts},
  author    = {Van Belle, Jente and Wen, Honglin and Verbeke, Wouter and Pinson, Pierre},
  year      = {2026},
  journal   = {arXiv preprint arXiv:2605.28531},
  url       = {https://arxiv.org/abs/2605.28531}
}
```

## Contact
Jente Van Belle (jente.vanbelle@kuleuven.be)
