# LOAD MODULES
# Standard library

# Third party
import torch
import numpy as np
import plotly.graph_objects as go

def centered_interval(center_value, lower_bound, num_points):

    # Calculate the number of points on each side of the center
    num_points_side = num_points // 2
    
    # Create the interval
    if num_points % 2 == 0:  # Even number of points
        interval_values = np.concatenate([
            np.linspace(lower_bound, center_value, num_points_side),
            np.linspace(center_value, lower_bound, num_points_side)
        ])
    else:  # Odd number of points
        interval_values = np.concatenate([
            np.linspace(lower_bound, center_value, num_points_side+1),
            np.linspace(center_value, lower_bound, num_points_side+1)[1:]
        ])
    
    return interval_values

def plot_quantile_forecasts(lookback_window, quantile_forecasts, forecast_period, sample, epoch):
    """
    Torch.Tensor input shapes:
    lookback_window = batch_size x backcast_length
    quantile_forecasts = batch_size x forecast_length x n_quantile_levels
    forecast_period = batch_size x forecast_length
    """
    while quantile_forecasts.shape[2] > 25:
        if quantile_forecasts.shape[2] % 2 == 0: # Even number of quantile_traces
            middle_index = quantile_forecasts.shape[2]//2
            if middle_index % 2 == 0:
                p1 = quantile_forecasts[..., :(middle_index):2]
                p2 = quantile_forecasts[..., (middle_index-1)].unsqueeze(-1)
                p3 = quantile_forecasts[..., (middle_index)].unsqueeze(-1)
                p4 = quantile_forecasts[..., (middle_index+1)::2]
                quantile_forecasts = torch.cat((p1, p2, p3, p4), dim=-1)
            else:
                p1 = quantile_forecasts[..., :(middle_index):2]
                p2 = quantile_forecasts[..., (middle_index)::2]
                quantile_forecasts = torch.cat((p1, p2), dim=-1)
        else: # Odd number of quantile_traces
            quantile_forecasts = quantile_forecasts[..., 0::2]
            
    lookback_periods = lookback_window.shape[1]
    _, forecast_periods, n_quantile_traces = quantile_forecasts.shape

    lookback_window = lookback_window[sample,:].numpy()
    quantile_forecasts = quantile_forecasts[sample,:,:].numpy()
    forecast_period = forecast_period[sample,:].numpy()

    fig = go.Figure()

    # Plot lookback window
    fig.add_trace(go.Scatter(x=np.arange(lookback_periods), 
                             y=lookback_window, 
                             mode='lines', name='Lookback Window', line=dict(color='black')))

    # Generate alphas for quantile traces
    alphas = centered_interval(1, 0.1, n_quantile_traces)

    # Create the custom color scale
    color_min = 'lightblue'  # Color for the minimum value
    color_center = 'darkblue'  # Center color
    color_max = 'lightblue'  # Color for the maximum value
    color_scale = [[0, color_min], [0.5, color_center], [1, color_max]]

    # Plot quantile forecasts
    for i in range(n_quantile_traces):
        quantile_trace = quantile_forecasts[:, i]
        fig.add_trace(go.Scatter(x=np.arange(lookback_periods, lookback_periods + forecast_periods), 
                                 y=quantile_trace,
                                 mode='lines', name=f'Quantile Trace {i}', opacity=alphas[i], line=dict(color='mediumblue')))

    # Plot forecast period
    fig.add_trace(go.Scatter(x=np.arange(lookback_periods, lookback_periods + forecast_periods), 
                             y=forecast_period, mode='lines', name='Forecast Period', line=dict(color='lawngreen', width=3)))

    if epoch is None:
        fig.update_layout(title=f"Best model",
                          xaxis_title='Periods',
                          yaxis_title='Values',
                          showlegend=False)
    else:
        fig.update_layout(title=f"Epoch {epoch}",
                          xaxis_title='Periods',
                          yaxis_title='Values',
                          showlegend=False)
    
    return fig
