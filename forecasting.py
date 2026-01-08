# forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class Forecasting:
    def __init__(self, df, model):
        self.df = df
        self.model = model
        self.forecasts = {}
    
    def create_forecast_features(self, steps=12):
        """Create features for forecasting"""
        # Use the most recent data point as starting point
        last_date = self.df.index[-1]
        last_ppi = self.df['PPIACO'].iloc[-1]
        last_ppi_change = self.df['PPI_change'].iloc[-1]
        last_ppi_yoy = self.df['PPI_yoy'].iloc[-1]
        
        # Generate future dates
        future_dates = pd.date_range(start=last_date, periods=steps+1, freq='M')[1:]
        
        # Simple forecast: assume PPI continues recent trend
        # This is a simplified approach - in practice, you'd want to forecast PPI first
        ppi_forecast = []
        current_ppi = last_ppi
        
        # Use average recent growth rate for PPI forecast
        recent_growth = self.df['PPI_change'].tail(6).mean() / 100  # Convert percentage to decimal
        
        for i in range(steps):
            current_ppi *= (1 + recent_growth)
            ppi_forecast.append(current_ppi)
        
        forecast_data = pd.DataFrame({
            'PPIACO': ppi_forecast,
            'PPI_lag1': [last_ppi] + ppi_forecast[:-1],  # Lagged values
            'PPI_change': [last_ppi_change] + [recent_growth * 100] * (steps-1),
            'PPI_yoy': [last_ppi_yoy] + [last_ppi_yoy] * (steps-1)  # Simplified
        }, index=future_dates)
        
        return forecast_data
    
    def forecast_cpi(self, steps=12):
        """Forecast future CPI values"""
        print(f"GENERATING {steps}-MONTH CPI FORECAST")
        print("=" * 40)
        
        # Create forecast features
        forecast_features = self.create_forecast_features(steps)
        
        # Prepare features for the model (same as training)
        features = ['PPI_lag1', 'PPI_change', 'PPI_yoy']
        X_forecast = forecast_features[features]
        
        # Make predictions
        cpi_forecast = self.model.predict(X_forecast)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'CPI_Forecast': cpi_forecast,
            'PPI_Forecast': forecast_features['PPIACO']
        }, index=forecast_features.index)
        
        # Calculate confidence intervals (simplified)
        # In practice, you'd use proper statistical methods for confidence intervals
        last_actual_cpi = self.df['CPIAUCSL'].iloc[-1]
        historical_volatility = self.df['CPIAUCSL'].pct_change().std() * 100
        
        forecast_df['CPI_Lower'] = forecast_df['CPI_Forecast'] * (1 - 0.01 * historical_volatility)
        forecast_df['CPI_Upper'] = forecast_df['CPI_Forecast'] * (1 + 0.01 * historical_volatility)
        
        print("Forecast Summary:")
        print(forecast_df[['CPI_Forecast', 'PPI_Forecast']].round(2))
        
        self.forecasts['cpi'] = forecast_df
        return forecast_df
    
    def plot_forecast(self, forecast_df, historical_months=24):
        """Plot the forecast along with historical data"""
        print("\nGENERATING FORECAST PLOT...")
        
        # Get recent historical data
        historical_data = self.df.tail(historical_months)
        
        plt.figure(figsize=(14, 8))
        
        # Plot historical CPI
        plt.plot(historical_data.index, historical_data['CPIAUCSL'], 
                label='Historical CPI', linewidth=2, color='blue')
        
        # Plot historical PPI (scaled to match CPI visually)
        ppi_scaled = historical_data['PPIACO'] * (historical_data['CPIAUCSL'].iloc[-1] / historical_data['PPIACO'].iloc[-1])
        plt.plot(historical_data.index, ppi_scaled, 
                label='Historical PPI (scaled)', linewidth=2, color='green', alpha=0.7)
        
        # Plot forecast
        plt.plot(forecast_df.index, forecast_df['CPI_Forecast'], 
                label='CPI Forecast', linewidth=2, color='red', linestyle='--')
        
        # Plot confidence interval
        plt.fill_between(forecast_df.index, 
                        forecast_df['CPI_Lower'], 
                        forecast_df['CPI_Upper'], 
                        alpha=0.2, color='red', label='Confidence Interval')
        
        # Add vertical line at forecast start
        forecast_start = forecast_df.index[0]
        plt.axvline(x=forecast_start, color='gray', linestyle=':', alpha=0.7)
        plt.text(forecast_start, plt.ylim()[0], 'Forecast Start', 
                rotation=90, verticalalignment='bottom')
        
        plt.title(f'CPI Forecast based on PPI\n({len(forecast_df)}-Month Outlook)')
        plt.xlabel('Date')
        plt.ylabel('Index Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add some statistics
        forecast_growth = (forecast_df['CPI_Forecast'].iloc[-1] / self.df['CPIAUCSL'].iloc[-1] - 1) * 100
        plt.figtext(0.02, 0.02, f'Projected {len(forecast_df)}-month CPI growth: {forecast_growth:.2f}%', 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.savefig('cpi_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt
    
    def calculate_forecast_metrics(self, test_periods=12):
        """Evaluate forecast accuracy on historical data"""
        print("\nFORECAST ACCURACY EVALUATION")
        print("=" * 40)
        
        if len(self.df) < test_periods * 2:
            print("Insufficient data for backtesting")
            return None
        
        # Use the last test_periods for evaluation
        test_data = self.df.tail(test_periods)
        train_data = self.df.iloc[:-test_periods]
        
        # Retrain model on training data
        features = ['PPI_lag1', 'PPI_change', 'PPI_yoy']
        X_train = train_data[features].dropna()
        y_train = train_data.loc[X_train.index, 'CPIAUCSL']
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Prepare test features
        X_test = test_data[features].dropna()
        y_test = test_data.loc[X_test.index, 'CPIAUCSL']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': np.mean(np.abs(y_test - y_pred)),
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        print("Backtesting Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics