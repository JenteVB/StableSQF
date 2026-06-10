# Install and load packages ###############################
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
required_packages <- list(
  data.table = "1.16.4",
  magrittr   = "2.0.3"
)
for (pkg in names(required_packages)) {
  if (!requireNamespace(pkg, quietly = TRUE) || 
      packageVersion(pkg) != required_packages[[pkg]]) {
    remotes::install_version(pkg, version = required_packages[[pkg]])
  }
}

library(data.table)
library(magrittr)

# Load data ###############################

# Specify dataset properties (see Baselines.R)
data_h <- 14 # 6 for M4M and 14 for M5_items
data_o <- 15 # 13 for M4M and 15 for M5_items
data_id_min <- 1
data_id_max <- 3049 # 48000 for M4M and 3049 for M5_items

# Quantile forecasts (qf) 
qf <- stop("Replace with SQF quantile forecasts")
# See main_*.py and WriteQuantileForecastsToCSV callback in src/utils/callbacks.py
# qf <- fread('M4_monthly_SQF_quantile_forecasts_test.csv')
# qf <- fread('M5_items_SQF_quantile_forecasts_test.csv')

qf[, origin := rep(rep(1:data_o, each = data_h), data_id_max-data_id_min+1)]
qf[, id := rep(data_id_min:data_id_max, each = data_h*data_o)]
qf[, h := horizon+1]
qf[, horizon := NULL]
setnames(qf, paste0("q", 1:100), as.character(seq(0.005, 0.995, 0.01)))

# Quantile forecasts lagged (qfl)
qfl <- stop("Replace with SQF quantile forecasts lagged")
# See main_*.py and WriteQuantileForecastsToCSV callback in src/utils/callbacks.py
# qfl <- fread('M4_monthly_SQF_quantile_forecasts_lagged_test.csv')
# qfl <- fread('M5_items_SQF_quantile_forecasts_lagged_test.csv')

qfl[, origin := rep(rep(1:data_o, each = data_h), data_id_max-data_id_min+1)]
qfl[, id := rep(data_id_min:data_id_max, each = data_h*data_o)]
qfl[, h := horizon+1]
qfl[, horizon := NULL]
setnames(qfl, paste0("q", 1:100), as.character(seq(0.005, 0.995, 0.01)))

# Wide-to-long reshaping and merging quantile forecasts and quantile forecast lagged
qf_long <- melt.data.table(qf,
                           id.vars = c('id', 'origin', 'h', 'actual', 'scaling_constant_abs', 'scaling_constant_sq'),
                           variable.name = 'quantile_level',
                           value.name = 'quantile_forecast')
rm(qf)
qf_long[, quantile_level := as.numeric(as.character(quantile_level))]
qfl_long <- melt.data.table(qfl,
                            id.vars = c('id', 'origin', 'h', 'actual', 'scaling_constant_abs', 'scaling_constant_sq'),
                            variable.name = 'quantile_level',
                            value.name = 'quantile_forecast_lagged')
rm(qfl)
qfl_long[, quantile_level := as.numeric(as.character(quantile_level))]
qfl_long[, h := h-1]
qf_all_long <- merge(qf_long, qfl_long, all.x = T)
rm(qf_long, qfl_long)

# SQF-stabilized - select Partial or Full ###############################
w_s <- 0.25
# 0.25 = lo
# 0.5 = med
# 0.75 = hi
# 1 = max

# # Partial
# qf_all_long[h < data_h, quantile_forecast := (1-w_s)*quantile_forecast+w_s*quantile_forecast_lagged]
# qf_all_long[, quantile_forecast_lagged_save := quantile_forecast_lagged]
# qf_all_long[, quantile_forecast_lagged := shift(quantile_forecast,(data_h-1)), by = .(id, quantile_level)]
# qf_all_long[origin == 1, quantile_forecast_lagged := quantile_forecast_lagged_save]
# qf_all_long[, quantile_forecast_lagged_save := NULL]
# qf_all_long[h == data_h, quantile_forecast_lagged := NA]

# Full
qf_all_long[, quantile_forecast_lagged_save := quantile_forecast_lagged]
for (o in 1:data_o) {
  qf_all_long[origin == o & h < data_h, quantile_forecast := (1-w_s)*quantile_forecast+w_s*quantile_forecast_lagged]
  qf_all_long[, quantile_forecast_lagged := shift(quantile_forecast, (data_h-1)), by = .(id, quantile_level)]
  qf_all_long[h == data_h, quantile_forecast_lagged := NA]
}
qf_all_long[origin == 1, quantile_forecast_lagged := quantile_forecast_lagged_save]
qf_all_long[, quantile_forecast_lagged_save := NULL]

# Forecast evaluation  ###############################

# Compute quantile scores
qf_all_long[, qs := (2*(as.numeric(actual <= quantile_forecast) - quantile_level)*
                       (quantile_forecast - actual))]

# Evaluation metrics
eval_metrics <- data.table(qf_all_long)
rm(qf_all_long)
quantile_levels <- unique(eval_metrics$quantile_level)
eval_metrics <- eval_metrics[
  , .(sCRPS = sum((qs/scaling_constant_abs)*(1/length(quantile_levels))),
      swcCRPS = sum((qs/scaling_constant_abs)*(quantile_level*(1-quantile_level))*(1/length(quantile_levels))),
      swtCRPS = sum((qs/scaling_constant_abs)*((2*quantile_level-1)^2)*(1/length(quantile_levels))),
      sW1 = sum((abs(quantile_forecast_lagged - quantile_forecast)/scaling_constant_abs)*(1/length(quantile_levels))),
      swcW1 = sum((abs(quantile_forecast_lagged - quantile_forecast)/scaling_constant_abs)*(quantile_level*(1-quantile_level))*(1/length(quantile_levels))),
      swtW1 = sum((abs(quantile_forecast_lagged - quantile_forecast)/scaling_constant_abs)*((2*quantile_level-1)^2)*(1/length(quantile_levels)))
      ),
  by = c('id', 'origin', 'h')]
eval_metrics$sCRPS %>% mean() %>% round(3)
eval_metrics$swcCRPS %>% mean() %>% round(3)
eval_metrics$swtCRPS %>% mean() %>% round(3)
eval_metrics$sW1 %>% mean(na.rm = T) %>% round(3)
eval_metrics$swcW1 %>% mean(na.rm = T) %>% round(3)
eval_metrics$swtW1 %>% mean(na.rm = T) %>% round(3)

