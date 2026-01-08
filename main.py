# main.py (updated with Step 3)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import load_and_preprocess_data, create_features
from exploratory_analysis import ExploratoryAnalysis
from regression_model import RegressionModel
from forecasting import Forecasting

print("PPI-CPI Analysis Project")
print("=" * 50)

def main():
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create additional features
    df_processed = create_features(df)
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"Columns: {list(df_processed.columns)}")
    
    # Step 2: Exploratory Data Analysis
    print("\nStep 2: Performing exploratory data analysis...")
    explorer = ExploratoryAnalysis(df_processed)
    exploratory_results = explorer.run_full_analysis()
    
    # Step 3: Regression Analysis
    print("\nStep 3: Performing regression analysis...")
    regression_analyzer = RegressionModel(df_processed)
    regression_models, regression_results = regression_analyzer.run_full_analysis()
    
    # Step 4: Forecasting
    print("\nStep 4: Generating forecasts...")
    
    # Use the linear regression model for forecasting
    if 'linear' in regression_models:
        linear_model = regression_models['linear']['model']
        forecaster = Forecasting(df_processed, linear_model)
        
        # Generate 12-month forecast
        forecast_df = forecaster.forecast_cpi(steps=12)
        
        # Plot the forecast
        forecaster.plot_forecast(forecast_df)
        
        # Evaluate forecast accuracy
        backtest_metrics = forecaster.calculate_forecast_metrics(test_periods=24)
    
    # Display final summary
    print("\n" + "="*70)
    print("PROJECT SUMMARY")
    print("="*70)
    
    # Correlation summary
    if hasattr(exploratory_results, 'get') and 'correlation_test' in exploratory_results:
        corr_coef, p_value = exploratory_results['correlation_test']
        print(f"PPI-CPI Correlation: {corr_coef:.4f} (p-value: {p_value:.4e})")
    
    # Best model performance
    if 'linear' in regression_models:
        test_r2 = regression_models['linear']['test_metrics']['R²']
        print(f"Best Model Test R²: {test_r2:.4f}")
        
        # Interpretation
        if test_r2 > 0.7:
            print("Model Interpretation: Strong relationship between PPI and CPI")
        elif test_r2 > 0.5:
            print("Model Interpretation: Moderate relationship between PPI and CPI")
        else:
            print("Model Interpretation: Weak relationship between PPI and CPI")
    
    print(f"\nAnalysis complete! Check generated plots:")
    print("- exploratory_analysis.png")
    print("- regression_results.png") 
    print("- cpi_forecast.png")
    
    # Save results to files
    df_processed.to_csv('processed_ppi_cpi.csv')
    print("\nProcessed data saved to 'processed_ppi_cpi.csv'")
    
    return df_processed, exploratory_results, regression_models, forecast_df

if __name__ == "__main__":
    results = main()