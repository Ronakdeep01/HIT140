from scipy.stats import mannwhitneyu

def test_risk_vs_rat_presence(df1):
    risk_with_rat = df1.loc[df1['rat_present_at_landing'] == 1, 'risk']
    risk_without_rat = df1.loc[df1['rat_present_at_landing'] == 0, 'risk']
    stat, p = mannwhitneyu(risk_with_rat, risk_without_rat)
    print(f"Mann–Whitney U test: statistic={stat:.2f}, p-value={p:.4f}")
    return stat, p

if __name__ == "__main__":
    print("✅ Statistical test module loaded successfully.")
