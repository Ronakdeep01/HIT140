import pandas as pd
import numpy as np

def feature_engineering(df1):
    df1['rat_present_at_landing'] = np.where(df1['seconds_after_rat_arrival'] >= 0, 1, 0)
    df1['season'] = df1['season'].replace({0:'winter', 1:'spring'})
    return df1

def missing_values_report(df):
    return df.isnull().sum()

if __name__ == "__main__":
    print("✅ Data analysis functions ready.")
