"""
bat_rat_analysis.py

Full pipeline for HIT140 Group Project - Investigations A & B.
Usage:
    python bat_rat_analysis.py

Adjust the DATA1_PATH and DATA2_PATH variables below to match your file locations.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# ----------------------------
# USER: update these paths if needed
DATA1_PATH = r"C:\Users\Al-Rayyan-Computer\OneDrive\Desktop\Bat Rat Project\bat_dataset1.csv"   # bat landings
DATA2_PATH = r"C:\Users\Al-Rayyan-Computer\OneDrive\Desktop\Bat Rat Project\bat_dataset2.csv"   # 30-min windows
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# Helper functions
def parse_mixed_datetime(s):
    """Parse mixed-format datetime strings robustly.
       Returns pd.NaT on failure.
    """
    if pd.isna(s):
        return pd.NaT
    # Try common formats first to speed up
    fmts = ["%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %I:%M", "%d/%m/%Y %I:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"]
    for f in fmts:
        try:
            return pd.to_datetime(s, format=f)
        except Exception:
            pass
    # fallback to dateutil parser
    try:
        return pd.to_datetime(parser.parse(str(s), dayfirst=True))
    except Exception:
        return pd.NaT

def safe_to_numeric(x):
    try:
        return pd.to_numeric(x)
    except Exception:
        return np.nan

# ----------------------------
# 1) Load data
print("Loading data...")
df1 = pd.read_csv(DATA1_PATH, dtype=str)   # read as strings first to handle messy cells
df2 = pd.read_csv(DATA2_PATH, dtype=str)

print(f"Raw df1 rows: {len(df1)}, df2 rows: {len(df2)}")

# ----------------------------
# 2) Clean & parse dataset1 (bat landing)
print("Cleaning dataset1...")
# Normalize column names (strip whitespace)
df1.columns = [c.strip() for c in df1.columns]

# show expected columns for df1
expected_cols_df1 = ['start_time','bat_landing_to_food','habit','rat_period_start','rat_period_end',
                     'seconds_after_rat_arrival','risk','reward','month','sunset_time','hours_after_sunset','season']
missing = [c for c in expected_cols_df1 if c not in df1.columns]
if missing:
    print("WARNING: expected columns missing from dataset1:", missing)

# Parse datetimes
for c in ['start_time','rat_period_start','rat_period_end','sunset_time']:
    if c in df1.columns:
        df1[c+'_parsed'] = df1[c].apply(parse_mixed_datetime)
    else:
        df1[c+'_parsed'] = pd.NaT

# Numeric conversions
df1['bat_landing_to_food'] = df1.get('bat_landing_to_food').apply(safe_to_numeric)
df1['seconds_after_rat_arrival'] = df1.get('seconds_after_rat_arrival').apply(safe_to_numeric)
df1['risk'] = df1.get('risk').apply(safe_to_numeric).astype('Int64')
df1['reward'] = df1.get('reward').apply(safe_to_numeric).astype('Int64')
df1['hours_after_sunset'] = df1.get('hours_after_sunset').apply(safe_to_numeric)
# month and season: keep as categorical
df1['month'] = df1.get('month')
df1['season'] = df1.get('season')

# create cleaned datetime column (fallback to start_time_parsed)
df1['start_dt'] = df1['start_time_parsed']

# If start_dt has NaT but month & sunset_time maybe present: skip - keep NaT
n_missing_start = df1['start_dt'].isna().sum()
print(f"dataset1: {n_missing_start} rows have unparsed start_time.")

# ----------------------------
# 3) Feature: was a rat present at bat landing?
# Several rows show rat_period_start and rat_period_end present. Use those when available.
def rat_present_at_landing(row):
    s = row['start_dt']
    r0 = row['rat_period_start_parsed']
    r1 = row['rat_period_end_parsed']
    if pd.isna(s):
        return np.nan
    if pd.notna(r0) and pd.notna(r1):
        return int((s >= r0) & (s <= r1))
    # fallback: seconds_after_rat_arrival - if negative => rat arrived after bat? In original dataset seconds_after_rat_arrival seems "since rats' arrival until bat landed"
    # The dataset's seconds_after_rat_arrival: positive means bat landed after rats arrival; so treat >=0 as rat present window
    sa = row.get('seconds_after_rat_arrival')
    try:
        if pd.notna(sa):
            # if seconds is huge (like > rat window?) we still mark present (zoologists labeled)
            return int(sa >= 0)
    except Exception:
        pass
    return 0

df1['rat_present'] = df1.apply(rat_present_at_landing, axis=1).astype('Int64')

# Save cleaned df1
df1.to_csv(OUT_DIR / "df1_cleaned.csv", index=False)
print("Saved df1_cleaned.csv")

# ----------------------------
# 4) Clean dataset2
print("Cleaning dataset2...")
df2.columns = [c.strip() for c in df2.columns]
expected_cols_df2 = ['time','month','hours_after_sunset','bat_landing_number','food_availability','rat_minutes','rat_arrival_number']
missing2 = [c for c in expected_cols_df2 if c not in df2.columns]
if missing2:
    print("WARNING: missing df2 columns:", missing2)

# parse time
df2['time_parsed'] = df2['time'].apply(parse_mixed_datetime)
# numeric conversions
for col in ['hours_after_sunset','bat_landing_number','food_availability','rat_minutes','rat_arrival_number']:
    if col in df2.columns:
        df2[col] = df2[col].apply(safe_to_numeric)

df2.to_csv(OUT_DIR / "df2_cleaned.csv", index=False)
print("Saved df2_cleaned.csv")

# ----------------------------
# 5) Quick EDA summaries
print("Running EDA summaries...")

# Basic counts
summary = {
    'df1_rows': len(df1),
    'df1_risk_count': int(df1['risk'].sum(skipna=True)) if 'risk' in df1.columns else None,
    'df1_rat_present_count': int(df1['rat_present'].sum(skipna=True)),
    'df2_rows': len(df2)
}
pd.Series(summary).to_csv(OUT_DIR / "summary_counts.csv")
print("Saved summary_counts.csv:", summary)

# Proportions of risk by rat_present
risk_by_rat = pd.crosstab(df1['rat_present'], df1['risk'], normalize='index') \
                .rename_axis(index='rat_present', columns='risk').reset_index()
risk_by_rat.to_csv(OUT_DIR / "risk_by_rat_presence.csv", index=False)
print("Saved risk_by_rat_presence.csv")

# Boxplot data: bat_landing_to_food by rat_present
boxdata = df1[['bat_landing_to_food','rat_present']].dropna()
boxdata.to_csv(OUT_DIR / "boxdata_batlanding_to_food_by_rat.csv", index=False)

# ----------------------------
# 6) Plots
print("Creating plots (PNG)...")
sns.set(style="whitegrid")

# 6.1 Proportion of risk when rats present/absent
plt.figure(figsize=(6,4))
ct = pd.crosstab(df1['rat_present'], df1['risk'], normalize='index') * 100
ct.plot(kind='bar', stacked=True)
plt.title("Percent distribution of risk by rat presence")
plt.xlabel("rat_present (0=absent, 1=present)")
plt.ylabel("percent")
plt.legend(title="risk")
plt.tight_layout()
plt.savefig(OUT_DIR / "risk_percent_by_rat_present.png", dpi=150)
plt.close()

# 6.2 Boxplot: bat_landing_to_food by rat presence
plt.figure(figsize=(6,4))
sns.boxplot(x='rat_present', y='bat_landing_to_food', data=df1)
plt.title("bat_landing_to_food by rat presence")
plt.xlabel("rat_present")
plt.ylabel("seconds")
plt.tight_layout()
plt.savefig(OUT_DIR / "box_batlanding_to_food_by_rat.png", dpi=150)
plt.close()

# 6.3 Seasonal boxplots - create season column if present
if 'season' in df1.columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='season', y='bat_landing_to_food', data=df1)
    plt.title("bat_landing_to_food by season")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "box_batlanding_to_food_by_season.png", dpi=150)
    plt.close()

# 6.4 df2: rat_arrival_number by season/month (if month present)
if 'month' in df2.columns:
    plt.figure(figsize=(10,4))
    sns.lineplot(x='time_parsed', y='rat_arrival_number', data=df2)
    plt.title("Time series: rat_arrival_number (df2)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "timeseries_rat_arrival_number.png", dpi=150)
    plt.close()

print("Plots saved to:", OUT_DIR.resolve())

# ----------------------------
# 7) Statistical tests - Investigation A
print("Statistical tests: Investigation A")

# 7.1 Compare bat_landing_to_food when rats present vs absent
grp_present = df1.loc[df1['rat_present']==1, 'bat_landing_to_food'].dropna().astype(float)
grp_absent  = df1.loc[df1['rat_present']==0, 'bat_landing_to_food'].dropna().astype(float)
print(f"n present={len(grp_present)}, n absent={len(grp_absent)}")

# Normality tests (Shapiro) - for reference
def safe_shapiro(arr):
    if len(arr) < 3:
        return (np.nan, np.nan)
    try:
        return stats.shapiro(arr)
    except Exception:
        return (np.nan, np.nan)

sh_p = safe_shapiro(grp_present)
sh_a = safe_shapiro(grp_absent)
print("Shapiro present:", sh_p, "Shapiro absent:", sh_a)

# Use Mann-Whitney U (non-parametric) by default
if len(grp_present) >= 1 and len(grp_absent) >= 1:
    try:
        u_stat, pval = stats.mannwhitneyu(grp_present, grp_absent, alternative='two-sided')
    except Exception:
        u_stat, pval = (np.nan, np.nan)
else:
    u_stat, pval = (np.nan, np.nan)

test_results = {
    'mannwhitney_u_stat': u_stat,
    'mannwhitney_pval': pval,
    'median_present': float(grp_present.median()) if len(grp_present)>0 else np.nan,
    'median_absent': float(grp_absent.median()) if len(grp_absent)>0 else np.nan
}
pd.Series(test_results).to_csv(OUT_DIR / "stat_tests_batlanding_to_food.csv")
print("Saved stat_tests_batlanding_to_food.csv", test_results)

# 7.2 Chi-square test: risk vs rat_present (binary)
ct = pd.crosstab(df1['risk'], df1['rat_present'])
ct.to_csv(OUT_DIR / "ct_risk_by_rat.csv")
try:
    chi2, chi_p, dof, expected = stats.chi2_contingency(ct.fillna(0))
except Exception:
    chi2, chi_p, dof, expected = (np.nan, np.nan, np.nan, None)

chi_res = {
    'chi2': chi2,
    'pvalue': chi_p,
    'dof': dof
}
pd.Series(chi_res).to_csv(OUT_DIR / "chi2_risk_vs_rat.csv")
print("Saved chi2_risk_vs_rat.csv", chi_res)

# ----------------------------
# 8) Logistic regression: model risk ~ rat_present + season + hours_after_sunset
print("Fitting logistic regression for risk...")

# prepare df with necessary columns and dropna
model_df = df1[['risk','rat_present','season','hours_after_sunset','month','bat_landing_to_food']].copy()
model_df = model_df.dropna(subset=['risk','rat_present'])  # must have risk and rat_present
# Convert types
model_df['risk'] = model_df['risk'].astype(int)
model_df['rat_present'] = model_df['rat_present'].astype(int)
# season as categorical
model_df['season'] = model_df['season'].astype('category')
# Fit logistic regression using statsmodels formula API
try:
    logit = smf.logit("risk ~ rat_present + C(season) + hours_after_sunset + bat_landing_to_food", data=model_df).fit(disp=False)
    print(logit.summary())
    with open(OUT_DIR / "logit_summary.txt", "w") as f:
        f.write(logit.summary().as_text())
except Exception as e:
    print("Logit failed:", e)
    logit = None

# Save model_df for review
model_df.to_csv(OUT_DIR / "logit_model_data.csv", index=False)
print("Saved logit_model_data.csv")

# ----------------------------
# 9) Investigation B — count data analysis (df2)
print("Investigation B: count models on dataset2 (rat arrivals)")

# Prepare df2 for modeling
df2_model = df2.dropna(subset=['rat_arrival_number','hours_after_sunset']).copy()
df2_model['rat_arrival_number'] = df2_model['rat_arrival_number'].astype(float)
# Optionally, create season mapping based on month number if available in df2['month']
# Here df2['month'] seems numeric categorical in your example.
try:
    df2_model['month'] = df2_model['month'].astype(int)
except Exception:
    pass

# Poisson GLM: rat_arrival_number ~ hours_after_sunset + bat_landing_number
# Note: rat_arrival_number contains small numbers including 0; Poisson appropriate as a start
df2_model['bat_landing_number'] = df2_model['bat_landing_number'].apply(safe_to_numeric)
df2_model = df2_model.dropna(subset=['rat_arrival_number','bat_landing_number'])

# Fit Poisson
try:
    poisson_model = smf.glm("rat_arrival_number ~ hours_after_sunset + bat_landing_number", 
                            data=df2_model, family=sm.families.Poisson()).fit()
    with open(OUT_DIR / "poisson_summary.txt", "w") as f:
        f.write(poisson_model.summary().as_text())
    print("Poisson GLM summary written.")
except Exception as e:
    print("Poisson model failed:", e)
    poisson_model = None

# ----------------------------
# 10) Interaction test: does rat_present effect on risk differ by season?
print("Testing interaction rat_present * season (logit)")

try:
    inter_logit = smf.logit("risk ~ rat_present * C(season) + hours_after_sunset + bat_landing_to_food", data=model_df).fit(disp=False)
    with open(OUT_DIR / "logit_interaction_summary.txt", "w") as f:
        f.write(inter_logit.summary().as_text())
    print("Saved interaction logit summary.")
except Exception as e:
    print("Interaction logit failed:", e)
    inter_logit = None

# ----------------------------
# 11) Save cleaned and processed datasets for reporting
df1.to_csv(OUT_DIR / "df1_processed.csv", index=False)
df2.to_csv(OUT_DIR / "df2_processed.csv", index=False)
print("Saved df1_processed.csv and df2_processed.csv")

print("All done. Check the 'output' folder for CSVs, plots and model summaries.")
