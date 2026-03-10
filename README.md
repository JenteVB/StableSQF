# *Stabilizing distribution-free probabilistic forecasts* </br><sub><sub>*Jente Van Belle* [[*Outlet Year*]](*url to paper*)</sub></sub>
*Multi-step-ahead forecasts are often updated as new observations become available, since shorter forecast horizons typically improve forecast quality. However, such improvements come at the cost of forecast instability, i.e., variability in forecasts for the same target period. This instability can trigger costly changes to plans formulated based on the forecasts and may erode trust in the forecasting system. In this work, we integrate forecast stability alongside forecast quality into the training of distribution-free probabilistic time-series forecasting models, allowing us to control this trade-off. We propose a method for generating stabilized forecasted conditional quantile functions using regression splines parameterized by a neural network. This approach enables joint optimization of stability and quality, as it allows us to directly penalize dissimilarities arising from forecast updates. Furthermore, it allows assigning varying importance to stabilizing different parts of the forecast distributions (e.g., central parts vs. tails) to focus on the parts most relevant for the intended downstream use (e.g., the upper tail for inventory management). We empirically evaluate the proposed method on two datasets with different statistical properties and show that it can effectively reduce forecast instability without a substantial loss in forecast quality and can target stabilization efforts toward specific parts of the forecast distributions.*

## Repository structure
This repository is organised as follows:
```bash
|- assets/
    |- *your own files*
|- config/
|- data/
|- lib/
|- notebooks/
|- res/
|- scripts/
|- src/
```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Citing
Please cite our paper and/or code as follows:
*Use the BibTeX citation*

```tex

@article{}

```
