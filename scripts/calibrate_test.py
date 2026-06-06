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

# ==============================================================================
# 2. TIME-SERIES CALIBRATION: MAXIMUM LIKELIHOOD ESTIMATION (MLE)
# ==============================================================================
def cir_neg_log_likelihood(params, r, dt):
    kappa, theta, sigma = params
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e10
    
    r_t = r[:-1]
    r_t_plus = r[1:]
    
    # Transition parameters
    c = 2 * kappa / (sigma**2 * (1 - np.exp(-kappa * dt)))
    df_df = 4 * kappa * theta / sigma**2
    nc = 2 * c * r_t * np.exp(-kappa * dt)
    
    # Avoid zero division or overflow
    # Transition probability follows a non-central chi-squared density for 2 * c * r_t_plus
    val = 2 * c * r_t_plus
    
    # Using scipy.stats.ncx2.pdf
    pdf = ncx2.pdf(val, df=df_df, nc=nc)
    
    # Check for zeros or NaNs in pdf to prevent log(0)
    pdf = np.maximum(pdf, 1e-12)
    
    # Log likelihood includes the Jacobian term 2*c
    log_lik = np.log(2 * c) + np.log(pdf)
    return -np.sum(log_lik)

res_mle = minimize(
    cir_neg_log_likelihood, 
    x0=[kappa_ols, theta_ols, sigma_ols], 
    bounds=[(1e-3, 10.0), (1e-3, 0.2), (1e-3, 1.0)],
    args=(r_train, dt), 
    method='L-BFGS-B'
)
kappa_mle, theta_mle, sigma_mle = res_mle.x
print("=== MLE Time-Series Calibration ===")
print(f"kappa: {kappa_mle:.6f}")
print(f"theta: {theta_mle:.6f}")
print(f"sigma: {sigma_mle:.6f}")
print(f"Feller Condition (2*kappa*theta >= sigma^2): {2*kappa_mle*theta_mle >= sigma_mle**2} ({2*kappa_mle*theta_mle:.6f} vs {sigma_mle**2:.6f})\n")

# ==============================================================================
# 3. CROSS-SECTIONAL CALIBRATION (TO FIT THE YIELD CURVE)
# ==============================================================================
def cir_yield(r, tau, kappa, theta, sigma):
    h = np.sqrt(kappa**2 + 2 * sigma**2)
    exp_h = np.exp(h * tau)
    
    B_num = 2 * (exp_h - 1)
    B_den = 2 * h + (kappa + h) * (exp_h - 1)
    B = B_num / B_den
    
    A_num = 2 * h * np.exp((kappa + h) * tau / 2)
    A_den = 2 * h + (kappa + h) * (exp_h - 1)
    A = (A_num / A_den) ** (2 * kappa * theta / sigma**2)
    
    # Yield formula
    yield_val = (B * r - np.log(A)) / tau
    return yield_val

def cross_sectional_loss(params, df, tenor_cols, maturities):
    kappa, theta, sigma = params
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e10
    
    r = df['ZC025YR'].values
    loss = 0
    for idx, col in enumerate(tenor_cols):
        # Actual yield
        y_act = df[col].values
        # Predicted yield
        y_pred = cir_yield(r, maturities[idx], kappa, theta, sigma)
        loss += np.mean((y_act - y_pred)**2)
    return loss

# Use OLS parameters as starting point
res_cs = minimize(
    cross_sectional_loss,
    x0=[kappa_ols, theta_ols, sigma_ols],
    bounds=[(1e-3, 5.0), (1e-3, 0.1), (1e-3, 0.5)],
    args=(train_df, tenor_cols, train_maturities),
    method='L-BFGS-B'
)
kappa_cs, theta_cs, sigma_cs = res_cs.x
print("=== Cross-Sectional Yield Curve Calibration ===")
print(f"kappa: {kappa_cs:.6f}")
print(f"theta: {theta_cs:.6f}")
print(f"sigma: {sigma_cs:.6f}")
print(f"Feller Condition (2*kappa*theta >= sigma^2): {2*kappa_cs*theta_cs >= sigma_cs**2} ({2*kappa_cs*theta_cs:.6f} vs {sigma_cs**2:.6f})\n")

# ==============================================================================
# EVALUATION ON TEST SET (MATURITIES UP TO 2 YEARS)
# ==============================================================================
def evaluate_model(kappa, theta, sigma, test_df, label):
    # Test set maturities: ZC025YR (3M), ZC050YR (6M), ZC075YR (9M), ZC100YR (1Y), ZC200YR (2Y)
    test_cols = ['ZC050YR', 'ZC075YR', 'ZC100YR', 'ZC200YR']
    test_maturities = np.array([0.5, 0.75, 1.0, 2.0])
    
    r_test = test_df['ZC025YR'].values
    
    y_actuals = []
    y_predicts = []
    
    for idx, col in enumerate(test_cols):
        y_act = test_df[col].values
        y_pred = cir_yield(r_test, test_maturities[idx], kappa, theta, sigma)
        
        y_actuals.extend(y_act)
        y_predicts.extend(y_pred)
        
    y_actuals = np.array(y_actuals)
    y_predicts = np.array(y_predicts)
    
    # Calculate R^2
    ss_res = np.sum((y_actuals - y_predicts)**2)
    ss_tot = np.sum((y_actuals - np.mean(y_actuals))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"Evaluation of {label} model on out-of-sample Test Set:")
    print(f"  Out-of-sample R^2: {r2:.4f}")
    return r2

evaluate_model(kappa_ols, theta_ols, sigma_ols, test_df, "OLS Time-Series")
evaluate_model(kappa_mle, theta_mle, sigma_mle, test_df, "MLE Time-Series")
evaluate_model(kappa_cs, theta_cs, sigma_cs, test_df, "Cross-Sectional")

