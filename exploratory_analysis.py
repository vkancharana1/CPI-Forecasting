# exploratory_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ExploratoryAnalysis:
    def __init__(self, df):
        self.df = df
        self.results = {}
    
    def basic_statistics(self):
        """Calculate basic descriptive statistics"""
        print("BASIC DESCRIPTIVE STATISTICS")
        print("=" * 50)
        
        stats_summary = self.df[['PPIACO', 'CPIAUCSL']].describe()
        print(stats_summary)
        
        # Additional statistics
        additional_stats = pd.DataFrame({
            'PPIACO': [
                stats.skew(self.df['PPIACO']),
                stats.kurtosis(self.df['PPIACO']),
                stats.variation(self.df['PPIACO'])*100
            ],
            'CPIAUCSL': [
                stats.skew(self.df['CPIAUCSL']),
                stats.kurtosis(self.df['CPIAUCSL']),
                stats.variation(self.df['CPIAUCSL'])*100
            ]
        }, index=['Skewness', 'Kurtosis', 'Coefficient of Variation (%)'])
        
        print("\nADDITIONAL STATISTICS")
        print(additional_stats)
        
        self.results['basic_stats'] = stats_summary
        self.results['additional_stats'] = additional_stats
        
        return stats_summary, additional_stats
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("\nCORRELATION ANALYSIS")
        print("=" * 50)
        
        # Pearson correlation
        pearson_corr = self.df[['PPIACO', 'CPIAUCSL']].corr(method='pearson')
        print("Pearson Correlation:")
        print(pearson_corr)
        
        # Spearman correlation (non-parametric)
        spearman_corr = self.df[['PPIACO', 'CPIAUCSL']].corr(method='spearman')
        print("\nSpearman Correlation:")
        print(spearman_corr)
        
        # Correlation test
        corr_coef, p_value = stats.pearsonr(self.df['PPIACO'], self.df['CPIAUCSL'])
        print(f"\nCorrelation Test: r={corr_coef:.4f}, p-value={p_value:.4e}")
        
        self.results['pearson_correlation'] = pearson_corr
        self.results['spearman_correlation'] = spearman_corr
        self.results['correlation_test'] = (corr_coef, p_value)
        
        return pearson_corr, spearman_corr
    
    def distribution_analysis(self):
        """Analyze distributions of PPI and CPI"""
        print("\nDISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Normality tests
        for col in ['PPIACO', 'CPIAUCSL']:
            stat, p_value = stats.normaltest(self.df[col])
            print(f"{col} - Normality test: statistic={stat:.4f}, p-value={p_value:.4e}")
            
            # Shapiro-Wilk test (for smaller datasets)
            if len(self.df) < 5000:
                shapiro_stat, shapiro_p = stats.shapiro(self.df[col])
                print(f"{col} - Shapiro-Wilk: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4e}")
        
        self.results['normality_tests'] = {
            'PPIACO': stats.normaltest(self.df['PPIACO']),
            'CPIAUCSL': stats.normaltest(self.df['CPIAUCSL'])
        }
    
    def create_visualizations(self):
        """Create exploratory visualizations"""
        print("\nCREATING VISUALIZATIONS...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PPI vs CPI - Exploratory Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time series plot
        axes[0, 0].plot(self.df.index, self.df['PPIACO'], label='PPI', linewidth=2)
        axes[0, 0].plot(self.df.index, self.df['CPIAUCSL'], label='CPI', linewidth=2)
        axes[0, 0].set_title('Time Series: PPI vs CPI')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Index Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot
        axes[0, 1].scatter(self.df['PPIACO'], self.df['CPIAUCSL'], alpha=0.6)
        axes[0, 1].set_title('Scatter Plot: CPI vs PPI')
        axes[0, 1].set_xlabel('PPI')
        axes[0, 1].set_ylabel('CPI')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add regression line
        z = np.polyfit(self.df['PPIACO'], self.df['CPIAUCSL'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.df['PPIACO'], p(self.df['PPIACO']), "r--", alpha=0.8)
        
        # 3. Histograms
        axes[0, 2].hist(self.df['PPIACO'], bins=30, alpha=0.7, label='PPI', density=True)
        axes[0, 2].hist(self.df['CPIAUCSL'], bins=30, alpha=0.7, label='CPI', density=True)
        axes[0, 2].set_title('Distribution Histograms')
        axes[0, 2].set_xlabel('Index Value')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Box plots
        data_to_plot = [self.df['PPIACO'], self.df['CPIAUCSL']]
        axes[1, 0].boxplot(data_to_plot, labels=['PPI', 'CPI'])
        axes[1, 0].set_title('Box Plots')
        axes[1, 0].set_ylabel('Index Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. QQ plots
        stats.probplot(self.df['PPIACO'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('PPI - Q-Q Plot')
        
        stats.probplot(self.df['CPIAUCSL'], dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('CPI - Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_full_analysis(self):
        """Run complete exploratory analysis"""
        print("RUNNING EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        self.basic_statistics()
        self.correlation_analysis()
        self.distribution_analysis()
        self.create_visualizations()
        
        return self.results