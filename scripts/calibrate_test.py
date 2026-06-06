import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ncx2

# Load cleaned data
train_df = pd.read_csv('data/train_data_cleaned.csv')
test_df = pd.read_csv('data/test_data_cleaned.csv')

# Pre-process dates
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

# Columns and maturities
tenor_cols = [c for c in train_df.columns if c != 'Date']
tenor_years = {'ZC025YR': 0.25, 'ZC050YR': 0.50, 'ZC075YR': 0.75, 'ZC100YR': 1.0,
               'ZC200YR': 2.0, 'ZC500YR': 5.0, 'ZC1000YR': 10.0, 'ZC2000YR': 20.0, 'ZC3000YR': 30.0}

train_maturities = np.array([tenor_years[col] for col in tenor_cols])

# 3M yield is our proxy for r_t
r_train = train_df['ZC025YR'].values
r_test = test_df['ZC025YR'].values

# Delta t is 1/252 (daily business days)
dt = 1 / 252

# ==============================================================================
# 1. TIME-SERIES CALIBRATION: ORDINARY LEAST SQUARES (OLS)
# ==============================================================================
def calibrate_ols(r, dt):
    # Discretized SDE: (r_{t+1} - r_t)/sqrt(r_t) = kappa * theta * dt / sqrt(r_t) - kappa * sqrt(r_t) * dt + sigma * sqrt(dt) * eps
    Y = np.diff(r) / np.sqrt(r[:-1])
    X1 = dt / np.sqrt(r[:-1])
    X2 = -np.sqrt(r[:-1]) * dt
    
    # Regression without intercept: Y = beta1*X1 + beta2*X2
    X = np.column_stack((X1, X2))
    beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    
    beta1, beta2 = beta[0], beta[1]
    kappa_ols = beta2
    theta_ols = beta1 / beta2
    
    # Residual variance yields sigma
    residuals = Y - (beta1 * X1 + beta2 * X2)
    sigma_ols = np.sqrt(np.var(residuals) / dt)
    
    return kappa_ols, theta_ols, sigma_ols

kappa_ols, theta_ols, sigma_ols = calibrate_ols(r_train, dt)
print("=== OLS Time-Series Calibration ===")
print(f"kappa: {kappa_ols:.6f}")
print(f"theta: {theta_ols:.6f}")
print(f"sigma: {sigma_ols:.6f}")
print(f"Feller Condition (2*kappa*theta >= sigma^2): {2*kappa_ols*theta_ols >= sigma_ols**2} ({2*kappa_ols*theta_ols:.6f} vs {sigma_ols**2:.6f})\n")

