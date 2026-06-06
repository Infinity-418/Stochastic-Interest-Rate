import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('data/train_data_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Define tenors in years for mapping
tenor_cols = [c for c in df.columns if c != 'Date']
tenor_years = {
    'ZC025YR': 0.25,
    'ZC050YR': 0.50,
    'ZC075YR': 0.75,
    'ZC100YR': 1.0,
    'ZC200YR': 2.0,
    'ZC500YR': 5.0,
    'ZC1000YR': 10.0,
    'ZC2000YR': 20.0,
    'ZC3000YR': 30.0
}
maturities = [tenor_years[col] for col in tenor_cols]

# 1. Term Structure: Average Yield Curve
mean_yields = df[tenor_cols].mean()
std_yields = df[tenor_cols].std()

plt.figure(figsize=(10, 5), dpi=100)
plt.plot(maturities, mean_yields * 100, 'o-', color='#1f77b4', linewidth=2, label='Mean Yield')
plt.fill_between(maturities, (mean_yields - std_yields)*100, (mean_yields + std_yields)*100, color='#1f77b4', alpha=0.15, label='±1 Std Dev')
plt.title('Average Yield Curve (Term Structure) - Training Data', fontsize=12, fontweight='bold')
plt.xlabel('Maturity (Years)', fontsize=10)
plt.ylabel('Yield (%)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('average_yield_curve.png')
plt.close()

# 2. Short Rate (3M) Evolution Over Time
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(df['Date'], df['ZC025YR'] * 100, color='#2ca02c', label='3-Month Yield (Short Rate Proxy)')
plt.title('Historical Evolution of the Short Rate (3M Yield)', fontsize=12, fontweight='bold')
plt.xlabel('Date', fontsize=10)
plt.ylabel('Yield (%)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('short_rate_evolution.png')
plt.close()

# 3. Correlation Heatmap
corr_matrix = df[tenor_cols].corr()
# Rename columns for presentation
corr_matrix.columns = [f"{tenor_years[c]}Y" for c in corr_matrix.columns]
corr_matrix.index = [f"{tenor_years[c]}Y" for c in corr_matrix.index]

plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".3f", cbar=True)
plt.title('Correlation Matrix of Different Maturities', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('yield_correlation_heatmap.png')
plt.close()

print("Exploratory Data Analysis completed! Generated plots:")
print("  - average_yield_curve.png")
print("  - short_rate_evolution.png")
print("  - yield_correlation_heatmap.png")
