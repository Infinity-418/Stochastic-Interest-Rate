import pandas as pd
import numpy as np

def clean_yield_data(file_path, output_path):
    print(f"Processing: {file_path}")
    # 1. Load data
    df = pd.read_csv(file_path)
    
    # 2. Clean column headers (strip whitespaces)
    df.columns = df.columns.str.strip()
    
    # 3. Format Date column and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 4. Outlier Detection & Normalisation (Spike-Reversal Filter)
    # We do this BEFORE reindexing to ensure single-day spikes aren't duplicated by forward-fill
    yield_cols = [col for col in df.columns if col != 'Date']
    cleaned_df = df.copy()
    outlier_count = 0
    
    for col in yield_cols:
        y = cleaned_df[col].values.astype(float)
        n = len(y)
        # Compute daily differences to find standard deviation of moves
        daily_diffs = np.diff(y)
        std_diff = np.std(daily_diffs)
        # Threshold: a daily move larger than 4 standard deviations of daily moves
        threshold = 4.0 * std_diff
        
        for i in range(1, n - 1):
            diff_prev = y[i] - y[i-1]
            diff_next = y[i] - y[i+1]
            
            # If both differences are large, have the same sign (spike up or down), 
            # and reverse immediately, it is a single-day error
            if (abs(diff_prev) > threshold and abs(diff_next) > threshold and 
                np.sign(diff_prev) == np.sign(diff_next)):
                # Replace outlier with linear interpolation of neighbors
                y[i] = 0.5 * (y[i-1] + y[i+1])
                outlier_count += 1
        
        cleaned_df[col] = y

    # 5. Handle Non-Trading Days (Reindex to Business Days)
    # This ensures a constant time step delta_t = 1/252 between consecutive observations
    cleaned_df = cleaned_df.set_index('Date')
    all_bdays = pd.date_range(start=cleaned_df.index.min(), end=cleaned_df.index.max(), freq='B')
    
    # Reindex and forward fill missing holidays/weekends, then backward fill if needed
    df_reindexed = cleaned_df.reindex(all_bdays)
    df_reindexed.index.name = 'Date'
    df_reindexed = df_reindexed.ffill().bfill().reset_index()
    
    print(f"  Reindexed to complete business days. Rows changed: {len(df)} -> {len(df_reindexed)}")
    print(f"  Outliers corrected: {outlier_count}")
    
    # 6. Save clean dataset
    df_reindexed.to_csv(output_path, index=False)
    print(f"  Saved clean file to: {output_path}\n")
    return df_reindexed

if __name__ == '__main__':
    train_cleaned = clean_yield_data('data/train_data.csv', 'data/train_data_cleaned.csv')
    test_cleaned = clean_yield_data('data/test_data.csv', 'data/test_data_cleaned.csv')
    test_3m_cleaned = clean_yield_data('data/test_data_3M.csv', 'data/test_data_3M_cleaned.csv')
