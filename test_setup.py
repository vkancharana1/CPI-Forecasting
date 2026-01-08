# test_setup.py
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    print("✓ All required packages imported successfully!")
    print("✓ Project setup is complete!")
except ImportError as e:
    print(f"✗ Import error: {e}")