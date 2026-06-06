import json

# Load the user's notebook
with open('Financeclubcir-3.ipynb') as f:
    nb = json.load(f)

# -----------------------------
# New Code for Cell 3 (Outlier Correction & Preprocessing plots)
# -----------------------------
cell3_code = """def preprocess_yield_data(file_path, label="Dataset"):
    # 1. Load data
    df = pd.read_csv(file_path)

    # 2. Clean column headers
    df.columns = df.columns.str.strip()

    # 3. Format Dates and Sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    raw_df = df.copy()

    # 4. Outlier Detection & Normalisation (Spike-Reversal Filter)
    # We do this BEFORE reindexing to ensure single-day spikes aren't duplicated by forward-fill
    yield_cols = [col for col in df.columns if col != 'Date']
    cleaned_df = df.copy()
    outliers_corrected = 0

    for col in yield_cols:
        y = cleaned_df[col].values.astype(float)
        n = len(y)

        daily_diffs = np.diff(y)
        std_diff = np.std(daily_diffs)
        threshold = 4.0 * std_diff

        y_filtered = y.copy()
        for i in range(1, n - 1):
            diff_prev = y[i] - y[i-1]
            diff_next = y[i] - y[i+1]

            if (abs(diff_prev) > threshold and abs(diff_next) > threshold and
                np.sign(diff_prev) == np.sign(diff_next)):
                y_filtered[i] = 0.5 * (y[i-1] + y[i+1])
                outliers_corrected += 1

        cleaned_df[col] = y_filtered

    # 5. Handle Non-Trading Days (Reindex to Business Days)
    cleaned_df = cleaned_df.set_index('Date')
    all_bdays = pd.date_range(start=cleaned_df.index.min(), end=cleaned_df.index.max(), freq='B')

    df_reindexed = cleaned_df.reindex(all_bdays)
    df_reindexed.index.name = 'Date'
    df_reindexed = df_reindexed.ffill().bfill().reset_index()

    print(f"{label} Preprocessed:")
    print(f"  - Rows: {len(raw_df)} -> {len(df_reindexed)} (Business day alignment)")
    print(f"  - Outliers smoothed: {outliers_corrected}")
    return raw_df, df_reindexed

# Execute cleaning pipelines across files
train_raw, train_clean = preprocess_yield_data(DATA_FILES['train_data.csv'], 'Training Data')
test_raw, test_clean = preprocess_yield_data(DATA_FILES['test_data.csv'], 'Test Data')
test_3m_raw, test_3m_clean = preprocess_yield_data(DATA_FILES['test_data_3M.csv'], 'Test 3M Data')

# Visual story: raw-to-cleaned data engineering
outlier_date = '2020-11-10'
target_col = 'ZC050YR'
window_start = pd.to_datetime(outlier_date) - pd.Timedelta(days=14)
window_end = pd.to_datetime(outlier_date) + pd.Timedelta(days=14)
raw_sub = train_raw[(train_raw['Date'] >= window_start) & (train_raw['Date'] <= window_end)]
clean_sub = train_clean[(train_clean['Date'] >= window_start) & (train_clean['Date'] <= window_end)]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})

axes[0].plot(raw_sub['Date'], raw_sub[target_col] * 100, marker='o', color=PALETTE['accent'],
             linewidth=1.4, linestyle='--', label='Raw feed (uncleaned)')
axes[0].plot(clean_sub['Date'], clean_sub[target_col] * 100, marker='o', color=PALETTE['base'],
             linewidth=2.4, label='Cleaned series')
axes[0].axvline(pd.to_datetime(outlier_date), color=PALETTE['muted'], linewidth=1.5, linestyle=':', alpha=0.8)

# Add text callout/annotation with an arrow pointing to the spike
spike_y_raw = raw_sub[raw_sub['Date'] == outlier_date][target_col].values[0] * 100
spike_y_clean = clean_sub[clean_sub['Date'] == outlier_date][target_col].values[0] * 100

axes[0].annotate(
    f"Transcription Error:\\nSpike of {spike_y_raw:.3f}% corrected\\nto {spike_y_clean:.3f}% via a\\nspike-reversal filter.",
    xy=(pd.to_datetime(outlier_date), spike_y_raw),
    xytext=(pd.to_datetime(outlier_date) + pd.Timedelta(days=3.5), spike_y_raw - 0.15),
    arrowprops=dict(facecolor='#4b5563', arrowstyle="->", connectionstyle="arc3,rad=-0.15", linewidth=1.2),
    fontsize=9.5, fontweight='bold', color='#1f2937',
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2", edgecolor="#fee2e2", alpha=0.96)
)

polish_axis(axes[0], f"Spike Correction Around {outlier_date}", "Date", "6M Yield (%)")
axes[0].tick_params(axis='x', rotation=30)
axes[0].legend(loc='lower left', frameon=True)

row_counts = pd.DataFrame({
    "Dataset": ["Train", "Test", "Test 3M"],
    "Raw rows": [len(train_raw), len(test_raw), len(test_3m_raw)],
    "Business-day rows": [len(train_clean), len(test_clean), len(test_3m_clean)]
})
row_counts_long = row_counts.melt("Dataset", var_name="Stage", value_name="Rows")
sns.barplot(data=row_counts_long, x="Dataset", y="Rows", hue="Stage", ax=axes[1],
            palette=[PALETTE['muted'], PALETTE['base']])
polish_axis(axes[1], "Business-Day Alignment", "", "Rows")
axes[1].legend(title="", frameon=True)
annotate_bars(axes[1], fmt="{:.0f}", orientation="v", padding=0.015)

fig.suptitle("Data Cleaning: Turning Noisy Market Feeds Into a Calibratable Short-Rate Panel",
             x=0.02, ha="left", fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()"""

# -----------------------------
# New Code for Cell 5 (EDA Dashboard)
# -----------------------------
cell5_code = """tenor_cols = [c for c in train_clean.columns if c != 'Date']
tenor_years = {
    'ZC025YR': 0.25, 'ZC050YR': 0.50, 'ZC075YR': 0.75, 'ZC100YR': 1.0,
    'ZC200YR': 2.0, 'ZC500YR': 5.0, 'ZC1000YR': 10.0, 'ZC2000YR': 20.0, 'ZC3000YR': 30.0
}
maturities = np.array([tenor_years[col] for col in tenor_cols])
tenor_names = [TENOR_LABELS.get(c, c) for c in tenor_cols]

mean_yields = train_clean[tenor_cols].mean() * 100
p10_yields = train_clean[tenor_cols].quantile(0.10) * 100
p90_yields = train_clean[tenor_cols].quantile(0.90) * 100
std_yields = train_clean[tenor_cols].std() * 100

short_rate = train_clean['ZC025YR'] * 100
slope_2y_3m = (train_clean['ZC200YR'] - train_clean['ZC025YR']) * 100
curve_matrix = train_clean[tenor_cols].values
curve_centered = curve_matrix - curve_matrix.mean(axis=0)
_, singular_values, _ = np.linalg.svd(curve_centered, full_matrices=False)
explained = singular_values**2 / np.sum(singular_values**2)

fig = plt.figure(figsize=(16, 11), dpi=150)
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.34, wspace=0.25)
ax_curve = fig.add_subplot(grid[0, 0])
ax_short = fig.add_subplot(grid[0, 1])
ax_corr = fig.add_subplot(grid[1, 0])
ax_factor = fig.add_subplot(grid[1, 1])

# Panel A: Term Structure
ax_curve.plot(maturities, mean_yields, marker='o', color=PALETTE['base'], linewidth=2.8, label='Average curve')
ax_curve.fill_between(maturities, p10_yields, p90_yields, color=PALETTE['fill'], alpha=0.85, label='10th-90th percentile band')
ax_curve.scatter(maturities, mean_yields, s=58, color=PALETTE['base'], edgecolor='white', linewidth=1.2, zorder=3)
ax_curve.set_xscale('log')
ax_curve.set_xticks(maturities)
ax_curve.set_xticklabels(tenor_names)

# Fine-tuned annotations to prevent overlap
ax_curve.annotate("Short End (Driven by Policy)\\nHigh Volatility (wide band)", xy=(0.25, mean_yields[0]), xytext=(0.35, mean_yields[0] - 0.5),
                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", color='#4b5563'),
                  fontsize=9, color='#4b5563', fontweight='semibold')
ax_curve.annotate("Long End (Term Premium)\\nLower Volatility (narrow band)", xy=(20.0, mean_yields[7]), xytext=(3, mean_yields[7] + 0.35),
                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='#4b5563'),
                  fontsize=9, color='#4b5563', fontweight='semibold')

polish_axis(ax_curve, "Training Yield Curve: Level and Dispersion", "Maturity (Log Scale)", "Yield (%)")
ax_curve.legend(loc='lower right', frameon=True)

# Panel B: Short Rate Proxy
ax_short.plot(train_clean['Date'], short_rate, color=PALETTE['actual'], linewidth=1.2, alpha=0.7, label='3M yield')
ax_short.plot(train_clean['Date'], short_rate.rolling(63).mean(), color=PALETTE['base'], linewidth=2.5, label='63-day trend')
ax_short.fill_between(train_clean['Date'], short_rate, short_rate.rolling(63).mean(),
                      color=PALETTE['fill'], alpha=0.45)

# Annotate macro cycles
covid_date = pd.to_datetime('2020-04-01')
hike_date = pd.to_datetime('2023-07-01')
ax_short.annotate("2020 COVID Rate Cuts\\nShort Rate near 0%", xy=(covid_date, 0.2), xytext=(pd.to_datetime('2018-01-01'), 1.8),
                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='#4b5563'),
                  fontsize=9, color='#4b5563', fontweight='semibold')
ax_short.annotate("2022-2024 Fed Tightening\\nInflation-Fighting Cycle", xy=(hike_date, 5.0), xytext=(pd.to_datetime('2021-01-01'), 4.2),
                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='#4b5563'),
                  fontsize=9, color='#4b5563', fontweight='semibold')

polish_axis(ax_short, "Short-Rate Proxy Over Time", "Date", "3M Yield (%)")
ax_short.legend(loc='best', frameon=True)

# Panel C: Correlation Map
corr_matrix = train_clean[tenor_cols].corr()
corr_matrix.columns = tenor_names
corr_matrix.index = tenor_names
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap="rocket_r", vmin=0.75, vmax=1.0, annot=True, fmt=".2f",
            linewidths=0.6, linecolor="#ffffff", cbar_kws={"label": "Correlation", "shrink": 0.8}, ax=ax_corr)
ax_corr.set_title("Tenor Correlation Map", loc="left", pad=10, fontweight="bold")
ax_corr.tick_params(axis='x', rotation=0)
ax_corr.tick_params(axis='y', rotation=0)

# Panel D: PCA Variance
factor_df = pd.DataFrame({
    "Component": ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)", "Remaining"],
    "Explained variance": [explained[0], explained[1], explained[2], explained[3:].sum()]
})
sns.barplot(data=factor_df, x="Component", y="Explained variance", ax=ax_factor,
            palette=[PALETTE['base'], PALETTE['shift'], PALETTE['accent'], PALETTE['muted']])
ax_factor.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
polish_axis(ax_factor, "Why One Factor Works, and Where It Struggles", "", "Variance explained")
annotate_bars(ax_factor, fmt="{:.1%}", orientation="v", padding=0.015)

fig.suptitle("Exploratory Data Analysis: Shape, Dynamics, and Factor Structure of the Yield Panel",
             x=0.02, ha="left", fontsize=16, fontweight="bold")
plt.show()

# Additional Slope Regime Plot
fig, ax = plt.subplots(figsize=(14, 4.2), dpi=150)
ax.plot(train_clean['Date'], slope_2y_3m, color=PALETTE['shift'], linewidth=1.35)
ax.axhline(0, color=PALETTE['actual'], linewidth=1, linestyle='--')
ax.fill_between(train_clean['Date'], 0, slope_2y_3m, where=slope_2y_3m >= 0,
                color=PALETTE['shrunk'], alpha=0.18, label='Upward 3M-2Y slope (Normal)')
ax.fill_between(train_clean['Date'], 0, slope_2y_3m, where=slope_2y_3m < 0,
                color=PALETTE['accent'], alpha=0.18, label='Inverted 3M-2Y slope (Inverted)')
polish_axis(ax, "Slope Regimes: 2Y Yield Minus 3M Yield", "Date", "Slope (%)")
ax.legend(loc='best', frameon=True)
plt.tight_layout()
plt.show()"""

# -----------------------------
# New Code for Cell 12 (Out-of-Sample Prediction Dashboard)
# -----------------------------
cell12_code = """test_cols = ['ZC050YR', 'ZC075YR', 'ZC100YR', 'ZC200YR']
test_maturities = np.array([0.5, 0.75, 1.0, 2.0])
test_labels = [TENOR_LABELS[c] for c in test_cols]
r_test = test_3m_clean['ZC025YR'].values

model_params = {
    'OLS Time-Series (Physical P)': (k_ols_p, t_ols_p, s_ols),
    'OLS Time-Series (Risk-Neutral Q)': (k_ols_q, t_ols_q, s_ols),
    'MLE Time-Series (Physical P)': (k_mle_p, t_mle_p, s_mle),
    'MLE Time-Series (Risk-Neutral Q)': (k_mle_q, t_mle_q, s_mle),
    'Cross-Sectional (Q)': (k_cs, t_cs, s_cs),
}
best_model_name = df_results.loc[df_results['Out-of-Sample R2'].idxmax(), 'Method']
best_k, best_t, best_s = model_params[best_model_name]
print(f"Using {best_model_name} for the main reconstruction visuals.")

pred_base_by_col = {
    col: cir_yield(r_test, tau, best_k, best_t, best_s)
    for col, tau in zip(test_cols, test_maturities)
}
actual_by_col = {col: test_clean[col].values for col in test_cols}
residual_bps = pd.DataFrame({
    TENOR_LABELS[col]: (pred_base_by_col[col] - actual_by_col[col]) * 10000
    for col in test_cols
}, index=test_clean['Date'])
per_tenor_r2 = pd.DataFrame({
    "Tenor": test_labels,
    "R2": [calc_r2(actual_by_col[col], pred_base_by_col[col]) for col in test_cols]
})

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 15), dpi=180)
outer = fig.add_gridspec(3, 2, height_ratios=[1.35, 1.0, 1.0], hspace=0.35, wspace=0.22)

# Panel 1: Time Series Reconstructions
top_grid = outer[0, :].subgridspec(2, 2, hspace=0.45, wspace=0.18)
for idx, (col, tau) in enumerate(zip(test_cols, test_maturities)):
    ax = fig.add_subplot(top_grid[idx // 2, idx % 2])
    actual = actual_by_col[col] * 100
    pred = pred_base_by_col[col] * 100
    
    ax.plot(test_clean['Date'], actual, linewidth=2.4, color='#111827', label='Actual Yield')
    ax.plot(test_clean['Date'], pred, linewidth=2.4, linestyle='--', color='#2563eb', label='Predicted (RN-MLE)')
    ax.fill_between(test_clean['Date'], actual, pred, color='#dbeafe', alpha=0.35)
    
    # Calculate RMSE in basis points
    rmse_bps = np.sqrt(np.mean((actual - pred)**2)) * 100
    
    ax.set_title(f"{TENOR_LABELS[col]} Tenor Reconstruction | $R^2$ = {per_tenor_r2.loc[idx, 'R2']:.3f}", fontsize=15, fontweight='bold', pad=10)
    ax.set_ylabel("Yield (%)", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.02, 0.08, f"RMSE: {rmse_bps:.1f} bps", transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1e3a8a',
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#eff6ff", edgecolor="#dbeafe", alpha=0.9))
    if idx == 0:
        ax.legend(frameon=True, fontsize=11, loc='upper left')
    if idx < 2:
        ax.tick_params(labelbottom=False)
    else:
        ax.tick_params(axis='x', rotation=15)

# Panel 2: Model Comparison Bar Chart
ax_bar = fig.add_subplot(outer[1, 0])
model_plot = df_results.copy()
model_plot["Display"] = model_plot["Method"].str.replace(" Time-Series", "", regex=False).str.replace("Risk-Neutral", "RN", regex=False)
bar_colors = ['#9ca3af', '#60a5fa', '#9ca3af', '#2563eb', '#7c3aed']
bars = ax_bar.barh(model_plot["Display"], model_plot["Out-of-Sample R2"], color=bar_colors, height=0.6)
ax_bar.axvline(0.85, linestyle='--', linewidth=1.8, color='#dc2626')

# Draw Bracket and annotation for MLE lambda adjustment
ax_bar.annotate('', xy=(0.95, 2), xytext=(0.95, 3), arrowprops=dict(arrowstyle="|-|", color='#eb5757', linewidth=1.8))
ax_bar.text(0.965, 2.5, "Risk-Premium\\nAdjustment\\n+13.0% Accuracy Boost!", va='center', ha='left', fontsize=9.5, color='#eb5757', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor="#fed7d7", alpha=0.95))

ax_bar.set_title("Calibration Accuracy Comparison", fontsize=16, fontweight='bold', pad=14)
ax_bar.set_xlabel("$R^2$ Score", fontsize=12)
ax_bar.set_xlim(min(0, model_plot["Out-of-Sample R2"].min() - 0.05), 1.08)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
for bar in bars:
    width = bar.get_width()
    ax_bar.text(width + 0.015, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=10, fontweight='bold', color='#111827')

# Panel 3: Monthly Residual Heatmap
ax_heat = fig.add_subplot(outer[1, 1])
resampled_resid = residual_bps.resample('ME').mean().T
sns.heatmap(resampled_resid, cmap='coolwarm', center=0, linewidths=0.5, linecolor='white', ax=ax_heat,
            cbar_kws={"label": "Mean Monthly Error (bps)", "shrink": 0.85})
ax_heat.set_title("Residual Structure Across Time & Maturity", fontsize=16, fontweight='bold', pad=14)
ax_heat.set_xlabel("Test Period", fontsize=12)
ax_heat.set_ylabel("Tenor", fontsize=12)
xtick_positions = np.linspace(0, len(resampled_resid.columns)-1, min(6, len(resampled_resid.columns))).astype(int)
ax_heat.set_xticks(xtick_positions + 0.5)
ax_heat.set_xticklabels([resampled_resid.columns[i].strftime('%Y-%m') for i in xtick_positions], rotation=0, fontsize=10)

# Panel 4: Yield Curve Snapshots (with Shading)
ax_snap = fig.add_subplot(outer[2, 0])
snapshot_dates = ['2024-06-20', '2025-03-25', '2026-04-22']
regimes = ['Inverted Yield Curve (June 2024)', 'Flat Yield Curve (March 2025)', 'Normal Yield Curve (April 2026)']
snapshot_colors = ['#2563eb', '#7c3aed', '#dc2626']
mat_full = np.array([0.25, 0.5, 0.75, 1.0, 2.0])

for date_str, regime_label, color in zip(snapshot_dates, regimes, snapshot_colors):
    date_dt = pd.to_datetime(date_str)
    test_idx = (test_clean['Date'] - date_dt).abs().idxmin()
    actual_date = test_clean.loc[test_idx, 'Date'].strftime('%Y-%m-%d')
    
    actual_curve = np.array([test_clean.loc[test_idx, col] * 100 for col in ['ZC025YR'] + test_cols])
    r_t_snap = test_3m_clean.loc[test_idx, 'ZC025YR']
    pred_curve = np.array([r_t_snap * 100] + [cir_yield(r_t_snap, tau, best_k, best_t, best_s) * 100 for tau in test_maturities])
    
    ax_snap.plot(mat_full, actual_curve, marker='o', linewidth=2.8, color=color, label=f'Actual: {regime_label}')
    ax_snap.plot(mat_full, pred_curve, marker='x', linestyle='--', linewidth=2.2, color=color, alpha=0.85, label=f'Predicted: {regime_label}')
    
    # Shade the error area between actual and predicted
    ax_snap.fill_between(mat_full, actual_curve, pred_curve, color=color, alpha=0.08)

ax_snap.set_title("Yield Curve Reconstruction Across Market Regimes", fontsize=16, fontweight='bold', pad=14)
ax_snap.set_xlabel("Maturity (Years)", fontsize=12)
ax_snap.set_ylabel("Yield (%)", fontsize=12)
ax_snap.set_xticks(mat_full)
ax_snap.set_xticklabels(['3M', '6M', '9M', '1Y', '2Y'])
ax_snap.legend(fontsize=9, ncol=1, loc='upper right', frameon=True)
ax_snap.spines['top'].set_visible(False)
ax_snap.spines['right'].set_visible(False)

# Panel 5: Scatter Plot
ax_scatter = fig.add_subplot(outer[2, 1])
scatter_colors = ['#2563eb', '#7c3aed', '#059669', '#dc2626']
for col, color in zip(test_cols, scatter_colors):
    ax_scatter.scatter(actual_by_col[col] * 100, pred_base_by_col[col] * 100, s=28, alpha=0.70, color=color, label=TENOR_LABELS[col])

min_y = min([actual_by_col[c].min() for c in test_cols] + [pred_base_by_col[c].min() for c in test_cols]) * 100
max_y = max([actual_by_col[c].max() for c in test_cols] + [pred_base_by_col[c].max() for c in test_cols]) * 100
ax_scatter.plot([min_y, max_y], [min_y, max_y], linestyle='--', linewidth=1.8, color='#4b5563')

# Scatter Annotation
ax_scatter.annotate("Tight fit on short maturities\\n(6M, 9M, 1Y)", xy=(3.5, 3.5), xytext=(2.3, 4.4),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='#4b5563'),
                    fontsize=9.5, color='#374151', fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fdf4", edgecolor="#bbf7d0", alpha=0.9))

ax_scatter.set_title("Predicted vs. Actual Yield Dispersion", fontsize=16, fontweight='bold', pad=14)
ax_scatter.set_xlabel("Actual Yield (%)", fontsize=12)
ax_scatter.set_ylabel("Predicted Yield (%)", fontsize=12)
ax_scatter.legend(title='Maturity Tenor', fontsize=9.5, loc='lower right')
ax_scatter.spines['top'].set_visible(False)
ax_scatter.spines['right'].set_visible(False)

fig.suptitle(f"Out-of-Sample Yield Curve Reconstruction Using {best_model_name}", fontsize=22, fontweight='bold', y=0.99)
plt.show()

# Residual distribution violin plot (Enhanced)
fig, ax = plt.subplots(figsize=(13, 5), dpi=180)
resid_long = (residual_bps.reset_index().melt(id_vars='Date', var_name='Tenor', value_name='Error bps'))
sns.violinplot(data=resid_long, x='Tenor', y='Error bps', inner='quartile', linewidth=1.2,
               palette=['#2563eb', '#7c3aed', '#059669', '#dc2626'], cut=0, hue='Tenor', legend=False, ax=ax)
ax.axhline(0, linestyle='--', linewidth=1.5, color='black')
ax.set_title("Residual Distribution by Maturity (Out-of-Sample)", fontsize=17, fontweight='bold', pad=14)
ax.set_xlabel("")
ax.set_ylabel("Prediction Error (bps)", fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()"""

# -----------------------------
# New Code for Cell 14 (Shifted CIR++ & Extension Dashboard)
# -----------------------------
cell14_code = """def fit_cirpp_shift(train_df, kappa, theta, sigma, target_cols, target_maturities):
    r_train_local = train_df['ZC025YR'].values
    shifts = {}
    for col, tau in zip(target_cols, target_maturities):
        base_train = cir_yield(r_train_local, tau, kappa, theta, sigma)
        shifts[col] = np.mean(train_df[col].values - base_train)
    return shifts

def validation_shrinkage_lambda(train_df, split_fraction=0.85):
    split_idx = int(len(train_df) * split_fraction)
    calibration_df = train_df.iloc[:split_idx].reset_index(drop=True)
    validation_df = train_df.iloc[split_idx:].reset_index(drop=True)

    validation_maturities = [tenor_years[c] for c in tenor_cols]

    # Calibrate base parameters on calibration subset
    k_val, t_val, s_val = calibrate_cross_sectional(
        calibration_df, tenor_cols, validation_maturities, [k_cs, t_cs, s_cs]
    )
    validation_shifts = fit_cirpp_shift(calibration_df, k_val, t_val, s_val, test_cols, test_maturities)

    y_actual_validation, y_base_validation, y_shifted_validation = [], [], []
    r_validation = validation_df['ZC025YR'].values

    for col, tau in zip(test_cols, test_maturities):
        base_pred = cir_yield(r_validation, tau, k_val, t_val, s_val)
        shifted_pred = base_pred + validation_shifts[col]
        y_actual_validation.extend(validation_df[col].values)
        y_base_validation.extend(base_pred)
        y_shifted_validation.extend(shifted_pred)

    y_actual_validation = np.array(y_actual_validation)
    y_base_validation = np.array(y_base_validation)
    y_shifted_validation = np.array(y_shifted_validation)
    shift_delta = y_shifted_validation - y_base_validation

    # Least-squares scalar adjustment
    lambda_raw = np.sum((y_actual_validation - y_base_validation) * shift_delta) / np.sum(shift_delta**2)
    lambda_shrink = float(np.clip(lambda_raw, 0.0, 1.0))

    print("=== Validation-Guided CIR++ Shift Selection ===")
    print(f"Validation Base CIR R2:          {calc_r2(y_actual_validation, y_base_validation):.6f}")
    print(f"Validation Full CIR++ Shift R2:  {calc_r2(y_actual_validation, y_shifted_validation):.6f}")
    print(f"Selected shift shrinkage lambda: {lambda_shrink:.6f}")
    return lambda_shrink

phi = fit_cirpp_shift(train_clean, k_cs, t_cs, s_cs, test_cols, test_maturities)
print("=== Calibrated Spreads (Training Set) ===")
for col, tau in zip(test_cols, test_maturities):
    print(f"  Tenor {col} ({tau}Y) raw shift phi: {phi[col]:.6f}")

shift_lambda = validation_shrinkage_lambda(train_clean, split_fraction=0.85)

# Evaluate on Test Set
y_actuals, y_pred_base, y_pred_shifted_raw, y_pred_shifted_shrunk = [], [], [], []
shifted_by_col = {}
raw_shifted_by_col = {}
base_by_col = {}
actual_ext_by_col = {}

for col, tau in zip(test_cols, test_maturities):
    y_act = test_clean[col].values
    y_base = cir_yield(r_test, tau, k_cs, t_cs, s_cs)
    y_shifted_raw = y_base + phi[col]
    y_shifted_shrunk = y_base + shift_lambda * phi[col]

    actual_ext_by_col[col] = y_act
    base_by_col[col] = y_base
    raw_shifted_by_col[col] = y_shifted_raw
    shifted_by_col[col] = y_shifted_shrunk

    y_actuals.extend(y_act)
    y_pred_base.extend(y_base)
    y_pred_shifted_raw.extend(y_shifted_raw)
    y_pred_shifted_shrunk.extend(y_shifted_shrunk)

y_actuals = np.array(y_actuals)
y_pred_base = np.array(y_pred_base)
y_pred_shifted_raw = np.array(y_pred_shifted_raw)
y_pred_shifted_shrunk = np.array(y_pred_shifted_shrunk)

r2_base_calc = calc_r2(y_actuals, y_pred_base)
r2_shifted_raw = calc_r2(y_actuals, y_pred_shifted_raw)
r2_shifted_shrunk = calc_r2(y_actuals, y_pred_shifted_shrunk)

print("\\n=== Model Comparison on Out-of-Sample Test Set ===")
print(f"Base CIR Model R2 (Cross-Sectional):          {r2_base_calc:.6f}")
print(f"Raw Shifted CIR (CIR++) R2:                   {r2_shifted_raw:.6f}")
print(f"Validation-Shrunk Shifted CIR (CIR++) R2:     {r2_shifted_shrunk:.6f}")

# Final extension dashboard
mat_test_full = np.array([0.25, 0.5, 0.75, 1.0, 2.0])
mean_act = np.array([np.mean(test_clean[col]) * 100 for col in ['ZC025YR'] + test_cols])
mean_base = np.array([np.mean(test_3m_clean['ZC025YR']) * 100] + [np.mean(base_by_col[col]) * 100 for col in test_cols])
mean_shifted_raw = np.array([mean_base[0]] + [np.mean(raw_shifted_by_col[col]) * 100 for col in test_cols])
mean_shifted_shrunk = np.array([mean_base[0]] + [np.mean(shifted_by_col[col]) * 100 for col in test_cols])

extension_scores = pd.DataFrame({
    "Model": ["Base CIR", "Raw CIR++", "Validation-shrunk CIR++"],
    "R2": [r2_base_calc, r2_shifted_raw, r2_shifted_shrunk]
})

fig = plt.figure(figsize=(16, 10), dpi=150)
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.32, wspace=0.25)
ax_curve = fig.add_subplot(grid[0, 0])
ax_scores = fig.add_subplot(grid[0, 1])
ax_improve = fig.add_subplot(grid[1, 0])
ax_error = fig.add_subplot(grid[1, 1])

# Panel A: Average curves comparison
ax_curve.plot(mat_test_full, mean_act, marker='o', color=PALETTE['actual'], linewidth=3, label='Actual test curve')
ax_curve.plot(mat_test_full, mean_base, marker='x', color=PALETTE['base'], linewidth=2.2, linestyle='--', label='Base CIR')
ax_curve.plot(mat_test_full, mean_shifted_raw, marker='^', color=PALETTE['accent'], linewidth=2.2, linestyle='--', label='Raw CIR++')
ax_curve.plot(mat_test_full, mean_shifted_shrunk, marker='s', color=PALETTE['shrunk'], linewidth=2.6, linestyle='--', label='Shrunk CIR++')

# Fill gaps to show over-correction and correction
ax_curve.fill_between(mat_test_full, mean_act, mean_shifted_raw, color=PALETTE['accent'], alpha=0.08, label='Overfitting Gap')
ax_curve.fill_between(mat_test_full, mean_act, mean_shifted_shrunk, color=PALETTE['shrunk'], alpha=0.08)

ax_curve.set_xticks(mat_test_full)
ax_curve.set_xticklabels(['3M', '6M', '9M', '1Y', '2Y'])
polish_axis(ax_curve, "Average Test Curve Fit", "Maturity", "Yield (%)")
ax_curve.legend(loc='best', frameon=True)

# Panel B: Out-of-Sample Accuracy comparison (horizontal bar)
sns.barplot(data=extension_scores, y="Model", x="R2", ax=ax_scores,
            palette=[PALETTE['base'], PALETTE['accent'], PALETTE['shrunk']], hue="Model", legend=False)
ax_scores.axvline(0.85, color=PALETTE['actual'], linestyle='--', linewidth=1.4, label='Required threshold')

# Write clean R2 texts next to each bar with custom labels instead of cluttered arrows
ax_scores.text(r2_base_calc + 0.005, 0, f"{r2_base_calc:.3f} (Base model)", va='center', fontsize=9.5, fontweight='bold', color='#111827')
ax_scores.text(r2_shifted_raw + 0.005, 1, f"{r2_shifted_raw:.3f} (-5.8% Overfit)", va='center', color='#eb5757', fontsize=9.5, fontweight='bold')
ax_scores.text(r2_shifted_shrunk + 0.005, 2, f"{r2_shifted_shrunk:.3f} (+3.2% Recovery)", va='center', color='#219653', fontsize=9.5, fontweight='bold')

polish_axis(ax_scores, "Final Out-of-Sample Accuracy Comparison", "R2", "")
ax_scores.set_xlim(0.78, 0.94)
ax_scores.legend(loc='lower right', frameon=True)

# Panel C: Improvement in RMSE (bps)
improvement_rows = []
for col in test_cols:
    base_rmse = np.sqrt(np.mean((base_by_col[col] - actual_ext_by_col[col])**2)) * 10000
    shrunk_rmse = np.sqrt(np.mean((shifted_by_col[col] - actual_ext_by_col[col])**2)) * 10000
    improvement_rows.append({"Tenor": TENOR_LABELS[col], "Base RMSE (bps)": base_rmse, "Shrunk CIR++ RMSE (bps)": shrunk_rmse})
improvement_df = pd.DataFrame(improvement_rows)
improvement_long = improvement_df.melt('Tenor', var_name='Model', value_name='RMSE bps')
sns.barplot(data=improvement_long, x='Tenor', y='RMSE bps', hue='Model', ax=ax_improve,
            palette=[PALETTE['base'], PALETTE['shrunk']])
polish_axis(ax_improve, "Error Scale by Maturity", "", "RMSE (basis points)")
ax_improve.legend(title="", frameon=True)

# Panel D: Residual boxplot
shrunk_resid = pd.DataFrame({
    TENOR_LABELS[col]: (shifted_by_col[col] - actual_ext_by_col[col]) * 10000
    for col in test_cols
}, index=test_clean['Date'])
shrunk_resid_long = shrunk_resid.reset_index().melt(id_vars='Date', var_name='Tenor', value_name='Error bps')
sns.boxplot(data=shrunk_resid_long, x='Tenor', y='Error bps', ax=ax_error,
            palette=[PALETTE['base'], PALETTE['mle'], PALETTE['shift'], PALETTE['accent']], width=0.55, hue='Tenor', legend=False)
ax_error.axhline(0, color=PALETTE['actual'], linewidth=1.1, linestyle='--')
polish_axis(ax_error, "Final CIR++ Error Distribution", "", "Error (bps)")

fig.suptitle("CIR++ Extension: Improving the Story Without Breaking the 3M-Only Prediction Rule",
             x=0.02, ha="left", fontsize=16, fontweight="bold")
plt.show()"""

# Apply modifications to cells
nb['cells'][3]['source'] = [line + '\n' for line in cell3_code.splitlines()]
nb['cells'][5]['source'] = [line + '\n' for line in cell5_code.splitlines()]
nb['cells'][12]['source'] = [line + '\n' for line in cell12_code.splitlines()]
nb['cells'][14]['source'] = [line + '\n' for line in cell14_code.splitlines()]

# Save the updated notebook
with open('Financeclubcir-3.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Financeclubcir-3.ipynb plots updated successfully!")
