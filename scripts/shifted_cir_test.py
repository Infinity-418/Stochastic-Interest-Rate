import pandas as pd
import numpy as np

# Load cleaned data
train_df = pd.read_csv('data/train_data_cleaned.csv')
test_df = pd.read_csv('data/test_data_cleaned.csv')

# Calibrated parameters from Cross-Sectional Calibration
kappa = 0.165644
theta = 0.024399
sigma = 0.001000

# Base CIR yield formula
def cir_yield(r, tau, kappa, theta, sigma):
    h = np.sqrt(kappa**2 + 2 * sigma**2)
    exp_h = np.exp(h * tau)
    B = 2 * (exp_h - 1) / (2 * h + (kappa + h) * (exp_h - 1))
    A = (2 * h * np.exp((kappa + h) * tau / 2) / (2 * h + (kappa + h) * (exp_h - 1))) ** (2 * kappa * theta / sigma**2)
    return (B * r - np.log(np.maximum(A, 1e-12))) / tau

# Evaluate Base CIR model on test set
test_cols = ['ZC050YR', 'ZC075YR', 'ZC100YR', 'ZC200YR']
test_maturities = [0.5, 0.75, 1.0, 2.0]

r_train = train_df['ZC025YR'].values
r_test = test_df['ZC025YR'].values

# Calculate tenor-specific shift phi_tau on the training set
phi = {}
print("=== Calibrating Shifted CIR (CIR++) ===")
for idx, col in enumerate(test_cols):
    tau = test_maturities[idx]
    y_act_train = train_df[col].values
    y_base_train = cir_yield(r_train, tau, kappa, theta, sigma)
    # The shift is the average residual between actual market yields and base CIR model yields
    phi[col] = np.mean(y_act_train - y_base_train)
    print(f"  Shift phi for tenor {col} ({tau}Y): {phi[col]:.6f}")

# Reconstruct yield curve on test set using base CIR vs shifted CIR
y_actuals = []
y_pred_base = []
y_pred_shifted = []

for idx, col in enumerate(test_cols):
    tau = test_maturities[idx]
    y_act = test_df[col].values
    y_base = cir_yield(r_test, tau, kappa, theta, sigma)
    y_shifted = y_base + phi[col]
    
    y_actuals.extend(y_act)
    y_pred_base.extend(y_base)
    y_pred_shifted.extend(y_shifted)

y_actuals = np.array(y_actuals)
y_pred_base = np.array(y_pred_base)
y_pred_shifted = np.array(y_pred_shifted)

# Calculate R^2
def calc_r2(act, pred):
    ss_res = np.sum((act - pred)**2)
    ss_tot = np.sum((act - np.mean(act))**2)
    return 1 - (ss_res / ss_tot)

r2_base = calc_r2(y_actuals, y_pred_base)
r2_shifted = calc_r2(y_actuals, y_pred_shifted)

print("\n=== Model Comparison on Out-of-Sample Test Set ===")
print(f"Base CIR Model R^2:    {r2_base:.6f}")
print(f"Shifted CIR (CIR++) R^2: {r2_shifted:.6f}")
