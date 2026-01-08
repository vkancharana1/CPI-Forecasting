# regression_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RegressionModel:
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
    
    def prepare_data(self, test_size=0.2, use_lags=True):
        """Prepare data for regression analysis"""
        print("PREPARING DATA FOR REGRESSION...")
        
        if use_lags and 'PPI_lag1' in self.df.columns:
            # Use lagged PPI to predict current CPI
            features = ['PPI_lag1', 'PPI_change', 'PPI_yoy']
            target = 'CPIAUCSL'
            
            # Remove rows with NaN (created by lagging and differencing)
            regression_data = self.df[features + [target]].dropna()
            
            X = regression_data[features]
            y = regression_data[target]
            
            print(f"Using features: {features}")
        else:
            # Simple regression: current PPI vs current CPI
            X = self.df[['PPIACO']]
            y = self.df['CPIAUCSL']
            print("Using simple PPI->CPI regression")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train, X_test
        self.X_train_scaled, self.X_test_scaled = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def simple_linear_regression(self):
        """Perform simple linear regression"""
        print("\nSIMPLE LINEAR REGRESSION")
        print("=" * 40)
        
        # Using scikit-learn
        lr = LinearRegression()
        lr.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred_train = lr.predict(self.X_train_scaled)
        y_pred_test = lr.predict(self.X_test_scaled)
        
        # Metrics
        train_metrics = self._calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self._calculate_metrics(self.y_test, y_pred_test)
        
        print("Training Metrics:")
        self._print_metrics(train_metrics)
        print("\nTest Metrics:")
        self._print_metrics(test_metrics)
        
        # Using statsmodels for detailed statistics
        X_sm = sm.add_constant(self.X_train_scaled)
        model_sm = sm.OLS(self.y_train, X_sm).fit()
        print("\nStatsModels Summary:")
        print(model_sm.summary())
        
        self.models['linear'] = {
            'model': lr,
            'model_sm': model_sm,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': {'train': y_pred_train, 'test': y_pred_test}
        }
        
        return lr, model_sm
    
    def regularized_regression(self):
        """Perform Ridge and Lasso regression"""
        print("\nREGULARIZED REGRESSION MODELS")
        print("=" * 40)
        
        # Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(self.X_train_scaled, self.y_train)
        
        y_pred_ridge_test = ridge.predict(self.X_test_scaled)
        ridge_metrics = self._calculate_metrics(self.y_test, y_pred_ridge_test)
        
        print("Ridge Regression Test Metrics:")
        self._print_metrics(ridge_metrics)
        
        # Lasso Regression
        lasso = Lasso(alpha=0.1)
        lasso.fit(self.X_train_scaled, self.y_train)
        
        y_pred_lasso_test = lasso.predict(self.X_test_scaled)
        lasso_metrics = self._calculate_metrics(self.y_test, y_pred_lasso_test)
        
        print("\nLasso Regression Test Metrics:")
        self._print_metrics(lasso_metrics)
        
        # Feature importance from Lasso
        if hasattr(lasso, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'coefficient': lasso.coef_,
                'abs_coefficient': np.abs(lasso.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("\nLasso Feature Importance:")
            print(feature_importance)
        
        self.models['ridge'] = {
            'model': ridge,
            'test_metrics': ridge_metrics,
            'predictions': y_pred_ridge_test
        }
        
        self.models['lasso'] = {
            'model': lasso,
            'test_metrics': lasso_metrics,
            'predictions': y_pred_lasso_test,
            'feature_importance': feature_importance
        }
        
        return ridge, lasso
    
    def cross_validation(self):
        """Perform cross-validation"""
        print("\nCROSS-VALIDATION RESULTS")
        print("=" * 40)
        
        lr = LinearRegression()
        
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(lr, self.X_train_scaled, self.y_train, 
                                  cv=5, scoring='r2')
        
        cv_rmse_scores = cross_val_score(lr, self.X_train_scaled, self.y_train,
                                       cv=5, scoring='neg_mean_squared_error')
        
        print(f"Cross-Validation R² Scores: {cv_scores}")
        print(f"Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"RMSE Scores: {np.sqrt(-cv_rmse_scores)}")
        print(f"Mean RMSE: {np.sqrt(-cv_rmse_scores).mean():.4f}")
        
        self.results['cross_validation'] = {
            'r2_scores': cv_scores,
            'rmse_scores': np.sqrt(-cv_rmse_scores)
        }
        
        return cv_scores
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'R²': r2_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _print_metrics(self, metrics):
        """Print metrics in formatted way"""
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    def plot_regression_results(self):
        """Plot regression results"""
        print("\nGENERATING REGRESSION PLOTS...")
        
        if 'linear' not in self.models:
            print("No linear model found. Run simple_linear_regression first.")
            return
        
        # Get predictions
        y_pred_test = self.models['linear']['predictions']['test']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regression Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(self.y_test, y_pred_test, alpha=0.6)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual CPI')
        axes[0, 0].set_ylabel('Predicted CPI')
        axes[0, 0].set_title('Actual vs Predicted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = self.y_test - y_pred_test
        axes[0, 1].scatter(y_pred_test, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Time series of actual vs predicted
        test_dates = self.y_test.index
        axes[1, 0].plot(test_dates, self.y_test, label='Actual CPI', linewidth=2)
        axes[1, 0].plot(test_dates, y_pred_test, label='Predicted CPI', linewidth=2, linestyle='--')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('CPI Value')
        axes[1, 0].set_title('Time Series: Actual vs Predicted')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Distribution of residuals
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add normal curve
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', lw=2)
        
        plt.tight_layout()
        plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_full_analysis(self):
        """Run complete regression analysis"""
        print("RUNNING REGRESSION ANALYSIS")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data()
        
        # Perform analyses
        self.simple_linear_regression()
        self.regularized_regression()
        self.cross_validation()
        self.plot_regression_results()
        
        # Summary
        print("\n" + "="*60)
        print("REGRESSION ANALYSIS SUMMARY")
        print("="*60)
        
        best_model = None
        best_r2 = -np.inf
        
        for name, model_data in self.models.items():
            if 'test_metrics' in model_data:
                r2 = model_data['test_metrics']['R²']
                print(f"{name.upper()} - Test R²: {r2:.4f}")
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = name
        
        print(f"\nBest model: {best_model} with R² = {best_r2:.4f}")
        
        return self.models, self.results