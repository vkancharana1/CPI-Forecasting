# data_loader.py
import pandas as pd
import numpy as np

def load_and_preprocess_data():
    """Load and preprocess the PPI-CPI data"""
    try:
        # Load the data
        df = pd.read_csv('PPI_CPI.csv')
        
        # Convert date column to datetime
        df['Observation_date'] = pd.to_datetime(df['Observation_date'])
        
        # Set date as index
        df.set_index('Observation_date', inplace=True)
        
        # Sort by date to ensure chronological order
        df.sort_index(inplace=True)
        
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    except FileNotFoundError:
        print("Error: PPI_CPI.csv file not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_features(df):
    """Create additional features for analysis"""
    # Calculate month-over-month changes
    df['PPI_change'] = df['PPIACO'].pct_change() * 100
    df['CPI_change'] = df['CPIAUCSL'].pct_change() * 100
    
    # Calculate year-over-year changes
    df['PPI_yoy'] = df['PPIACO'].pct_change(12) * 100
    df['CPI_yoy'] = df['CPIAUCSL'].pct_change(12) * 100
    
    # Create lag features for PPI (previous month's PPI)
    df['PPI_lag1'] = df['PPIACO'].shift(1)
    
    # Remove rows with NaN values created by differencing
    df_clean = df.dropna()
    
    return df_clean