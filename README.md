# Stochastic Interest Rate Modelling and Prediction
### Cox-Ingersoll-Ross Calibration and Validation-Shrunk CIR++ Extension
*Finance Club, IIT Roorkee — Open Projects 2026*

---

I picked the Cox-Ingersoll-Ross (CIR) model over Vasicek because I wanted to guarantee rate positivity. The 2020–2022 pandemic period had such weird rate behavior that Vasicek felt like it would blow up. While Vasicek is simpler, CIR's state-dependent volatility is much better suited to capture the massive shift between the near-zero rate regime and the recent tightening cycle. The goal here was to calibrate this model to historical treasury yields and see if I could reconstruct the out-of-sample yield curve (from 6M out to 2Y tenors) using only the 3M yield as the model input.

---

## Wrong Turns and What Failed

It wasn't a straight line to get here. My baseline Ordinary Least Squares (OLS) calibration estimated a negative mean reversion speed ($\kappa^{\mathbb{P}} \approx -0.2439$). This was the first sign something was off with using the physical measure directly for bond pricing—an explosive SDE rate path makes no sense for valuation. 

Additionally, my first run of the Shifted CIR++ model out-of-sample was a disaster. The $R^2$ dropped from 89.18% to 83.39%. The model was overfitting the training set's low-rate era. I had to write a validation loop to shrink the shifts ($\lambda_{\text{shrink}} \approx 0.5882$) to rescue the extension.

---

## Calibration and Reconstruction Results

Models are evaluated out-of-sample on the test set (2024–2026). The out-of-sample $R^2$ scores are summarized below:

| Model Calibration | $\kappa^{\mathbb{Q}}$ | $\theta^{\mathbb{Q}}$ | $\sigma$ | Feller Passed? | Out-of-Sample $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| OLS Time-Series (Physical $\mathbb{P}$) | $-0.2439$ | $-0.0052$ | $0.0395$ | Yes | `0.8016` (Clipped) |
| OLS Time-Series (Risk-Neutral $\mathbb{Q}$) | $+0.0470$ | $+0.0269$ | $0.0395$ | Yes | `0.8304` |
| MLE Time-Series (Physical $\mathbb{P}$) | $+0.3385$ | $+0.0377$ | $0.0452$ | Yes | `0.8171` |
| MLE Time-Series (Risk-Neutral $\mathbb{Q}$) | **$+0.5463$** | **$+0.0234$** | **$0.0452$** | **Yes** | **`0.9467`** |
| Cross-Sectional (Q) | **$+0.1656$** | **$+0.0244$** | **$0.0015$** | **Yes** | **`0.8918`** |
| Validation-Shrunk CIR++ | — | — | — | — | **`0.8661`** |

The MLE model under the risk-neutral measure $\mathbb{Q}$ was the clear winner, easily outperforming the baseline time-series models. Physical parameter models ($\mathbb{P}$) failed to achieve the target $R^2$ threshold, whereas transitioning to the risk-neutral measure $\mathbb{Q}$ using the Market Price of Risk ($\lambda$) shifted the MLE out-of-sample accuracy to 94.67%.

---

## Data Preprocessing

The raw data had a few issues worth noting. The preprocessing pipeline:
* **Strips whitespace:** Cleans column headers.
* **Filters outlier spikes:** Detects single-day yield jumps/drops exceeding $4\sigma$ of daily moves that immediately reverse. Running this filter *before* reindexing ensures that holiday gaps (like Veterans Day on November 11, 2020) do not duplicate the spike via forward-filling. This successfully corrects the November 10, 2020 outlier from `0.406%` to `0.149%`.
* **Aligns to business days:** Reindexes to a business-day calendar (`freq='B'`) and forward-fills weekends/market holidays ($\Delta t = 1/252$ year).

---

## File Structure

```
├── .gitignore
├── Problem_statement.pdf
├── README.md
├── Stochastic Interest Rate_modelling.ipynb
├── data
│   ├── Problem_statement.pdf
│   ├── test_data.csv
│   ├── test_data_3M.csv
│   ├── test_data_3M_cleaned.csv
│   ├── test_data_cleaned.csv
│   ├── train_data.csv
│   └── train_data_cleaned.csv
├── plots
│   ├── average_yield_curve.png
│   ├── dashboard_test.png
│   ├── extension_test.png
│   ├── notebook_plot_1.png
│   ├── notebook_plot_2.png
│   ├── notebook_plot_3.png
│   ├── notebook_plot_4.png
│   ├── notebook_plot_5.png
│   ├── notebook_plot_6.png
│   ├── short_rate_evolution.png
│   └── yield_correlation_heatmap.png
└── scripts
    ├── calibrate_test.py
    ├── clean_data.py
    ├── eda.py
    ├── shifted_cir_test.py
    ├── test_plot_generator.py
    └── update_notebook_graphs.py
```

---

## Instructions for Execution

### Google Colab
1. Upload **`Stochastic Interest Rate_modelling.ipynb`** to Google Drive and open it.
2. Select **Runtime > Run all**. The notebook downloads the datasets and runs automatically.

### Local Terminal
Configure the python environment:
```bash
pip install numpy pandas scipy matplotlib seaborn
```
* **Execute Data Cleaning:** `python3 scripts/clean_data.py`
* **Generate dashboards locally:** `python3 scripts/test_plot_generator.py`
* **Inject dashboard code to notebook cells:** `python3 scripts/update_notebook_graphs.py`

---

## Appendix: Mathematical Details
For full mathematical derivations, transition density formulas, and analytical pricing derivations, please refer to the attached **[Problem_statement.pdf](./Problem_statement.pdf)** or review the markdown cells in **[Stochastic Interest Rate_modelling.ipynb](./Stochastic%20Interest%20Rate_modelling.ipynb)**.
