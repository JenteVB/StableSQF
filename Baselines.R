# Install and load packages ###############################
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
required_packages <- list(
  forecast   = "8.23.0",
  data.table = "1.16.4",
  magrittr   = "2.0.3",
  doParallel = "1.0.17",
  stringr    = "1.5.1"
)
for (pkg in names(required_packages)) {
  if (!requireNamespace(pkg, quietly = TRUE) || 
      packageVersion(pkg) != required_packages[[pkg]]) {
    remotes::install_version(pkg, version = required_packages[[pkg]])
  }
}

library(forecast)
library(data.table)
library(magrittr)
library(doParallel)
library(stringr)

# Load data ###############################

# M4 monthly
M4S_train <- fread('https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv',
                   fill = T)
M4S_test  <- fread('https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv')
setnames(M4S_train, names(M4S_train), c('id', 1:(ncol(M4S_train)-1)))
setnames(M4S_test, names(M4S_test), c('id', 1:(ncol(M4S_test)-1)))
M4S <- merge.data.table(M4S_train, M4S_test, by = 'id')
M4S[, id := str_remove(id, 'M') %>% as.numeric()]
setnames(M4S, names(M4S), c('id', 1:(ncol(M4S_train)+ncol(M4S_test)-2)))
M4S <- melt.data.table(M4S, id.vars = 'id')
M4S <- M4S[!is.na(value)]
setorder(M4S, variable)
M4S[, variable:= c(1:.N) , by = 'id']
M4S <- data.frame(M4S)
M4S_freq <- 12
M4S_h <- 6
M4S_o <- 13
M4S_oh <- 18

# M5 items
M5_items <- fread("https://raw.githubusercontent.com/rakshitha123/StableForecasting/master/datasets/m5_items.txt")
setnames(M5_items, names(M5_items), as.character(1:(ncol(M5_items))))
M5_items[, id := .I]
M5_items <- melt.data.table(M5_items, id.vars = 'id')
setorder(M5_items, variable)
# Remove leading zeros per id
M5_items <- M5_items[, {
  first_nonzero_idx <- which(value != 0)[1]
  if (is.na(first_nonzero_idx)) .SD[0] else .SD[first_nonzero_idx:.N]
}, by = id]
setorder(M5_items, variable)
M5_items[, variable := NULL]
M5_items[, variable := c(1:.N), by = 'id']
M5_items <- data.frame(M5_items)
M5_items_freq <- 7
M5_items_h <- 14
M5_items_o <- 15
M5_items_oh <- 28

# Summary statistics M4 monthly ###############################
# ts_lengths
M4S <- data.table(M4S)
M4S_ts_lengths <- M4S[, .N, by = id]
M4S_ts_lengths$N %>% min()
M4S_ts_lengths$N %>% median()
M4S_ts_lengths$N %>% max()
# CV^2 where ts is non-zero
M4S_cv2 <- M4S[, {
  x <- value
  nzx <- which(x != 0)
  x <- x[nzx]
  if (length(x) > 1 && mean(x) != 0) {
    .(cv2 = (sd(x) / mean(x))^2)
  } else {
    .(cv2 = NA)
  }
}, by = id]
M4S_cv2$cv2 %>% min()
M4S_cv2$cv2 %>% median()
M4S_cv2$cv2 %>% max()
# ADI
M4S_adi <- M4S[, {
  x <- value
  nzx <- which(x != 0)
  k <- length(nzx)
  x <- c(nzx[1], nzx[2:k] - nzx[1:(k - 1)])
  .(adi = mean(x))
}, by = id]
M4S_adi$adi %>% min()
M4S_adi$adi %>% median()
M4S_adi$adi %>% max()
M4S[, mean(value), by = id]$V1 %>% min()
M4S[, mean(value), by = id]$V1 %>% median()
M4S[, mean(value), by = id]$V1 %>% max()
M4S[, sd(value), by = id]$V1 %>% min()
M4S[, sd(value), by = id]$V1 %>% median()
M4S[, sd(value), by = id]$V1 %>% max()
M4S_SBA <- merge.data.table(M4S_cv2, M4S_adi)
nrow(M4S_SBA[cv2 <= 0.49 & adi <= 1.32])/nrow(M4S_SBA) # smooth
nrow(M4S_SBA[cv2 > 0.49 & adi <= 1.32])/nrow(M4S_SBA) # irregular
nrow(M4S_SBA[cv2 <= 0.49 & adi > 1.32])/nrow(M4S_SBA) # intermittent
nrow(M4S_SBA[cv2 > 0.49 & adi > 1.32])/nrow(M4S_SBA) # lumpy

# Summary statistics M5 items ###############################
# ts_lengths
M5_items <- data.table(M5_items)
M5_items_ts_lengths <- M5_items[, .N, by = id]
M5_items_ts_lengths$N %>% min()
M5_items_ts_lengths$N %>% median()
M5_items_ts_lengths$N %>% max()
# CV^2 where ts is non-zero
M5_items_cv2 <- M5_items[, {
  x <- value
  nzx <- which(x != 0)
  x <- x[nzx]
  if (length(x) > 1 && mean(x) != 0) {
    .(cv2 = (sd(x) / mean(x))^2)
  } else {
    .(cv2 = NA)
  }
}, by = id]
M5_items_cv2$cv2 %>% min()
M5_items_cv2$cv2 %>% median()
M5_items_cv2$cv2 %>% max()
# ADI
M5_items_adi <- M5_items[, {
  x <- value
  nzx <- which(x != 0)
  k <- length(nzx)
  x <- c(nzx[1], nzx[2:k] - nzx[1:(k - 1)])
  .(adi = mean(x))
}, by = id]
M5_items_adi$adi %>% min()
M5_items_adi$adi %>% median()
M5_items_adi$adi %>% max()
M5_items[, mean(value), by = id]$V1 %>% min()
M5_items[, mean(value), by = id]$V1 %>% median()
M5_items[, mean(value), by = id]$V1 %>% max()
M5_items[, sd(value), by = id]$V1 %>% min()
M5_items[, sd(value), by = id]$V1 %>% median() 
M5_items[, sd(value), by = id]$V1 %>% max()
M5_items_SBA <- merge.data.table(M5_items_cv2, M5_items_adi)
nrow(M5_items_SBA[cv2 <= 0.49 & adi <= 1.32])/nrow(M5_items_SBA) # smooth
nrow(M5_items_SBA[cv2 > 0.49 & adi <= 1.32])/nrow(M5_items_SBA) # irregular
nrow(M5_items_SBA[cv2 <= 0.49 & adi > 1.32])/nrow(M5_items_SBA) # intermittent
nrow(M5_items_SBA[cv2 > 0.49 & adi > 1.32])/nrow(M5_items_SBA) # lumpy

# Select data and choose validation/test mode  ###############################

data <- M4S
data_freq <- M4S_freq
data_h <- M4S_h
data_o <- M4S_o
data_oh <- M4S_oh
data_id_min <- 1
data_id_max <- data$id %>% unique() %>% length()
data_lbw_multiplier <- 8 # 8 for M4S and 26 for M5_items
rm(M4S, M4S_train, M4S_test)

# data <- M5_items
# data_freq <- M5_items_freq
# data_h <- M5_items_h
# data_o <- M5_items_o
# data_oh <- M5_items_oh
# data_id_min <- 1
# data_id_max <- data$id %>% unique() %>% length()
# data_lbw_multiplier <- 26 # 8 for M4S and 26 for M5_items
# rm(M5_items)

VAL_MODE <- FALSE # TRUE or FALSE

# Forecast generation ###############################
forecast_method <- 'ETS-G' # 'ETS-G', 'ETS-B', 'mean-G', 'mean-B', 'snaive-G', or 'snaive-B'

ncores <- detectCores()
cl <- makeCluster(ncores, outfile = "")
registerDoParallel(cl)

pb <- txtProgressBar(min = data_id_min, max = data_id_max, style = 3)

quantile_forecasts_df <- foreach(id = c(data_id_min:data_id_max), 
                                 .combine = 'rbind', 
                                 .packages = 'forecast') %dopar% {
                                       
    ts <- data[data$id == id,]
    ts <- ts[order(ts$variable),]$value
    if (VAL_MODE == T) ts <- ts[1:(length(ts)-data_oh)]
                                       
    quantile_forecasts_id <- data.frame()
    for (origin in c(0:data_o)) {
      
      ts_origin <- ts(ts[1:(length(ts)-(data_h+data_o)+origin)],
                      frequency = data_freq)
      ts_actuals <- ts[(length(ts)-(data_h+data_o)+origin+1):(length(ts)-(data_h+data_o)+origin+data_h)]
      
      if (forecast_method == 'ETS-G') {
        fit <- ets(ts_origin)
        forecasts <- forecast(fit, h = data_h,
                              PI = TRUE,
                              simulate = FALSE, bootstrap = FALSE,
                              level = c(seq(0.01, 0.99, 0.02)))
      } else if (forecast_method == 'ETS-B') {
        fit <- ets(ts_origin)
        forecasts <- forecast(fit, h = data_h,
                              PI = TRUE,
                              simulate = TRUE, bootstrap = TRUE, npaths = 5000,
                              level = c(seq(0.01, 0.99, 0.02)))
      } else if (forecast_method == 'mean-G') {
        forecasts <- meanf(tail(ts_origin, data_lbw_multiplier*data_h), h = data_h,
                           bootstrap = FALSE,
                           level = c(seq(0.01, 0.99, 0.02)))
      } else if (forecast_method == 'mean-B') {
        forecasts <- meanf(tail(ts_origin, data_lbw_multiplier*data_h), h = data_h,
                           bootstrap = TRUE, npaths = 5000,
                           level = c(seq(0.01, 0.99, 0.02)))
      } else if (forecast_method == 'snaive-G') {
        forecasts <- snaive(ts_origin, h = data_h,
                            bootstrap = FALSE,
                            level = c(seq(0.01, 0.99, 0.02)))
      } else if (forecast_method == 'snaive-B') {
        forecasts <- snaive(ts_origin, h = data_h,
                            bootstrap = TRUE, npaths = 5000,
                            level = c(seq(0.01, 0.99, 0.02)))
      }
      
      quantile_forecasts <- cbind(as.data.frame(forecasts$lower[, rev(seq_len(ncol(forecasts$lower)))]),
                                  as.data.frame(forecasts$upper))
      colnames(quantile_forecasts) <- c(
        "0.5%", "1.5%", "2.5%", "3.5%", "4.5%", "5.5%", "6.5%", "7.5%", "8.5%", "9.5%", 
        "10.5%", "11.5%", "12.5%", "13.5%", "14.5%", "15.5%", "16.5%", "17.5%", "18.5%", 
        "19.5%", "20.5%", "21.5%", "22.5%", "23.5%", "24.5%", "25.5%", "26.5%", "27.5%", 
        "28.5%", "29.5%", "30.5%", "31.5%", "32.5%", "33.5%", "34.5%", "35.5%", "36.5%", 
        "37.5%", "38.5%", "39.5%", "40.5%", "41.5%", "42.5%", "43.5%", "44.5%", "45.5%", 
        "46.5%", "47.5%", "48.5%", "49.5%", "50.5%", "51.5%", "52.5%", "53.5%", "54.5%", 
        "55.5%", "56.5%", "57.5%", "58.5%", "59.5%", "60.5%", "61.5%", "62.5%", "63.5%", 
        "64.5%", "65.5%", "66.5%", "67.5%", "68.5%", "69.5%", "70.5%", "71.5%", "72.5%", 
        "73.5%", "74.5%", "75.5%", "76.5%", "77.5%", "78.5%", "79.5%", "80.5%", "81.5%", 
        "82.5%", "83.5%", "84.5%", "85.5%", "86.5%", "87.5%", "88.5%", "89.5%", "90.5%", 
        "91.5%", "92.5%", "93.5%", "94.5%", "95.5%", "96.5%", "97.5%", "98.5%", "99.5%"
      )
      
      quantile_forecasts$id <- rep(id, data_h)
      quantile_forecasts$origin <- rep(origin, data_h)
      quantile_forecasts$h <- c(1:data_h)
      quantile_forecasts$actual <- ts_actuals
      quantile_forecasts$scaling_constant_abs <- rep(mean(abs(diff(ts_origin))), data_h) + 1e-3
      quantile_forecasts$scaling_constant_sq <- rep(mean(diff(ts_origin)^2), data_h) + 1e-3
      quantile_forecasts_id <- rbind(quantile_forecasts_id,
                                     quantile_forecasts)
      
    } # for loop end -- origins
    
    setTxtProgressBar(pb, id)
    #output
    return(quantile_forecasts_id)
    
  } # foreach loop end -- time series

stopCluster(cl)
registerDoSEQ()

# Forecast evaluation  ###############################

# Wide-to-long reshaping
quantile_forecasts_dt <- data.table(quantile_forecasts_df)
quantile_forecasts_dt <- melt.data.table(
  quantile_forecasts_dt,
  id.vars = c('id', 'origin', 'h', 'actual', 'scaling_constant_abs', 'scaling_constant_sq'),
  variable.name = 'quantile_level',
  value.name = 'quantile_forecast')

# Create lagged values for stability calculations
setorder(quantile_forecasts_dt, origin, h)
quantile_forecasts_dt[, quantile_forecast_lagged := shift(quantile_forecast, data_h-1), by = c('id', 'quantile_level')]
quantile_forecasts_dt[h == 6, quantile_forecast_lagged := NA]
quantile_forecasts_dt <- quantile_forecasts_dt[origin > 0]

# Replace negative values
quantile_forecasts_dt[quantile_forecast < 0, quantile_forecast := 0]
quantile_forecasts_dt[quantile_forecast_lagged < 0, quantile_forecast_lagged := 0]
quantile_forecasts_dt[, quantile_level := as.numeric(str_remove(quantile_level, '%'))/100]

# Compute quantile scores
quantile_forecasts_dt[, qs := (2*(as.numeric(actual <= quantile_forecast) - quantile_level)*
                                 (quantile_forecast - actual))]
# # An alternative (but equivalent) calculation
# quantile_forecasts_dt[, qs := (2*((quantile_level*pmax(actual - quantile_forecast, 0)) +
#                                     ((1-quantile_level)*pmax(quantile_forecast - actual, 0))))]

# Evaluation metrics
eval_metrics <- data.table(quantile_forecasts_dt)
rm(quantile_forecasts_dt)
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

