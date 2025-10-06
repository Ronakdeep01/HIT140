import pandas as pd
import numpy as np

def normalize_column_names(df):
    orig_cols = df.columns.tolist()
    new_cols = []
    for c in orig_cols:
        nc = str(c).strip().lower()
        nc = "".join([ch if ch.isalnum() else "_" for ch in nc])
        while "__" in nc:
            nc = nc.replace("__", "_")
        new_cols.append(nc)
    df.columns = new_cols
    return df

def try_parse_dates(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True)
    return df

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

if __name__ == "__main__":
    print("✅ Cleaning module loaded successfully.")
