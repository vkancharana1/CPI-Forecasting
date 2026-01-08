# 📊 PPI–CPI Relationship Analysis & Inflation Forecasting

## 📌 Project Overview

This project analyzes the **relationship between Producer Price Index (PPI) and Consumer Price Index (CPI)** and builds a **data-driven framework** to:

* Explore statistical properties of inflation indicators
* Quantify the strength of the PPI → CPI relationship
* Build regression models to explain CPI movements
* Generate **forward CPI forecasts** using PPI-based predictors

The project follows a **modular, production-style structure** commonly used in quantitative research and economic analytics.

---

## 🎯 Objectives

* Understand how **producer-level inflation transmits to consumer inflation**
* Perform **robust exploratory data analysis (EDA)**
* Build and compare **linear, Ridge, and Lasso regression models**
* Evaluate model performance using **out-of-sample testing & cross-validation**
* Forecast CPI over a **12-month horizon**
* Produce **publication-quality visualizations**

---

## 🧠 Key Concepts Covered

* Inflation economics (PPI vs CPI)
* Time-series feature engineering (lags, MoM & YoY changes)
* Correlation & distribution analysis
* Regression diagnostics & residual analysis
* Regularization (Ridge & Lasso)
* Forecasting & backtesting

---

## 🛠️ Technologies & Tools Used

* **Python**
* **Pandas & NumPy** – data manipulation
* **Matplotlib & Seaborn** – visualization
* **SciPy & StatsModels** – statistical testing
* **Scikit-learn** – regression modeling & validation

---

## 📂 Project Structure

```
PPI-CPI-Analysis/
│
├── main.py                     # End-to-end execution pipeline
├── data_loader.py              # Data loading & feature engineering
├── exploratory_analysis.py     # Statistical & visual EDA
├── regression_model.py         # Linear, Ridge & Lasso regression
├── forecasting.py              # CPI forecasting & backtesting
├── utils.py                    # Helper & statistical utilities
├── test_setup.py               # Environment & dependency checks
├── PPI_CPI.csv                 # Raw input data
│
├── exploratory_analysis.png    # EDA visual outputs
├── regression_results.png      # Regression diagnostics
├── cpi_forecast.png            # CPI forecast visualization
└── README.md
```

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA module performs:

* **Descriptive statistics** (mean, variance, skewness, kurtosis)
* **Correlation analysis**

  * Pearson correlation
  * Spearman rank correlation
* **Normality tests**

  * D’Agostino–Pearson
  * Shapiro-Wilk
* **Visualizations**

  * Time-series plots
  * Scatter plots with regression fit
  * Histograms & box plots
  * Q-Q plots

📈 Output saved as `exploratory_analysis.png`.

---

## 📐 Regression Modeling

The regression module evaluates:

### Models Implemented

* **Linear Regression**
* **Ridge Regression**
* **Lasso Regression**

### Features Used

* Lagged PPI (`PPI_lag1`)
* Month-over-Month PPI change
* Year-over-Year PPI change

### Evaluation Metrics

* R²
* RMSE
* MAE
* MAPE
* Cross-validated R² & RMSE

📊 Includes:

* Actual vs Predicted plots
* Residual diagnostics
* Feature importance (Lasso)

📈 Output saved as `regression_results.png`.

---

## 🔮 Forecasting

The forecasting module uses the **best-performing regression model** to:

* Generate **12-month CPI forecasts**
* Construct **confidence intervals**
* Overlay historical CPI & PPI trends
* Perform **backtesting** on historical periods

📈 Output saved as `cpi_forecast.png`.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/PPI-CPI-Analysis.git
cd PPI-CPI-Analysis
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

### 3️⃣ Run the Project

```bash
python main.py
```

The script will:

* Load and preprocess data
* Run EDA
* Train regression models
* Generate forecasts
* Save plots and processed datasets

---

## 📈 Outputs Generated

* `exploratory_analysis.png`
* `regression_results.png`
* `cpi_forecast.png`
* `processed_ppi_cpi.csv`

---

## 📚 What I Learned

* How inflation indicators interact across economic layers
* Translating economic intuition into **quantitative features**
* Building **modular, scalable Python research pipelines**
* Interpreting regression diagnostics in an economic context
* Forecast evaluation and backtesting techniques

---
