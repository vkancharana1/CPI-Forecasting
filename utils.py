# utils.py
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae
    }
    
    return metrics

def print_metrics(metrics_dict, title="Model Metrics"):
    """Print metrics in a formatted way"""
    print(f"\n{title}")
    print("-" * 30)
    for metric, value in metrics_dict.items():
        print(f"{metric}: {value:.4f}")

def check_stationarity(timeseries):
    """Basic stationarity check using differencing"""
    from statsmodels.tsa.stattools import adfuller
    
    # Perform Augmented Dickey-Fuller test
    result = adfuller(timeseries.dropna())
    
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    return result[1] < 0.05  # Return True if stationary