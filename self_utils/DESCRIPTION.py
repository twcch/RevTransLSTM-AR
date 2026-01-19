import os
import pandas as pd
import numpy as np
from scipy.stats import jarque_bera
from utils.ADFtest import calculate_ADF

DATA_DIR = "data"
MAX_FILES = 5
ALPHA = 0.05  # significance level
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "descriptive_stats.csv")

def describe_file(path, save_per_file=True):
    df = pd.read_csv(path)
    cols = list(df.columns)
    if "date" in cols:
        cols_no_date = [c for c in cols if c != "date"]
        df_num = df[cols_no_date].select_dtypes(include=[np.number])
    else:
        cols_no_date = cols
        df_num = df.select_dtypes(include=[np.number])

    if df_num.shape[1] == 0:
        print(f"Skip {os.path.basename(path)}: no numeric columns\n")
        return pd.DataFrame()  # empty

    # Try to get ADF results (calculate_ADF returns adfuller tuples)
    try:
        adf_results = calculate_ADF(os.path.dirname(path) + "/", os.path.basename(path))
        adf_map = {col: res for col, res in zip(cols_no_date, adf_results)}
    except Exception:
        adf_map = {}

    rows = []
    for col in df_num.columns:
        series = df_num[col].dropna()
        n = series.shape[0]
        mean = series.mean()
        std = series.std(ddof=0)
        min_v = series.min()
        max_v = series.max()
        skewness = series.skew()
        fisher_kurt = series.kurtosis()  # excess kurtosis
        pearson_kurt = fisher_kurt + 3.0

        try:
            jb_stat, jb_p = jarque_bera(series)
        except Exception:
            jb_stat, jb_p = np.nan, np.nan

        adf_res = adf_map.get(col)
        if adf_res is not None:
            adf_tstat = adf_res[0]
            adf_p = adf_res[1]
        else:
            adf_tstat = np.nan
            adf_p = np.nan

        jb_reject = (jb_p < ALPHA) if not np.isnan(jb_p) else None
        non_stationary = (adf_p > ALPHA) if not np.isnan(adf_p) else None

        rows.append({
            "file": os.path.basename(path),
            "column": col,
            "Obs": n,
            "Mean": mean,
            "Std Dev": std,
            "Min": min_v,
            "Max": max_v,
            "Skewness": skewness,
            "Kurtosis": pearson_kurt,
            "J-B Stat": jb_stat,
            "J-B p": jb_p,
            "J-B reject normal": jb_reject,
            "ADF t-stat": adf_tstat,
            "ADF p": adf_p,
            "Non-stationary (ADF p > 0.05)": non_stationary
        })

    df_out = pd.DataFrame(rows)

    # reorder columns so the main ones come first
    desired_order = [
        "file", "column", "Obs", "Mean", "Std Dev", "Min", "Max",
        "Skewness", "Kurtosis", "J-B Stat", "J-B p", "J-B reject normal",
        "ADF t-stat", "ADF p", "Non-stationary (ADF p > 0.05)"
    ]
    # keep only existing columns in order
    cols_to_save = [c for c in desired_order if c in df_out.columns]
    df_out = df_out[cols_to_save]

    if save_per_file:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        per_file = os.path.join(RESULTS_DIR, f"per_{os.path.basename(path)}_stats.csv")
        df_out.to_csv(per_file, index=False)
        print(f"Saved per-file stats: {per_file}")

    return df_out

def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])
    if not files:
        print("No CSV files found in data/")
        return

    all_frames = []
    for fname in files[:MAX_FILES]:
        df_stats = describe_file(os.path.join(DATA_DIR, fname), save_per_file=True)
        if not df_stats.empty:
            all_frames.append(df_stats)

            # quick summary per file: % non-stationary columns
            non_stat = df_stats['Non-stationary (ADF p > 0.05)'].apply(lambda x: bool(x) if pd.notna(x) else False)
            pct_nonstat = non_stat.mean() * 100
            print(f"{fname}: {pct_nonstat:.1f}% columns non-stationary (ADF p > {ALPHA})")

    if all_frames:
        all_df = pd.concat(all_frames, ignore_index=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        all_df.to_csv(RESULTS_FILE, index=False)
        print(f"Saved aggregated stats: {RESULTS_FILE}")
    else:
        print("No numeric columns found in the processed files.")

if __name__ == "__main__":
    main()