import os
from src.1_data_loading import load_datasets
from src.2_data_cleaning import normalize_column_names, try_parse_dates, safe_numeric
from src.3_data_analysis import feature_engineering
from src.4_visualization import visualize
from src.5_statistical_tests import test_risk_vs_rat_presence

OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df1, df2 = load_datasets()

df1 = normalize_column_names(df1)
df2 = normalize_column_names(df2)
df1 = try_parse_dates(df1, ['start_time','rat_period_start','rat_period_end','sunset_time'])
df2 = try_parse_dates(df2, ['time'])

df1 = safe_numeric(df1, ['bat_landing_to_food','seconds_after_rat_arrival','risk','reward','hours_after_sunset'])
df2 = safe_numeric(df2, ['hours_after_sunset','bat_landing_number','food_availability','rat_minutes','rat_arrival_number'])

df1 = feature_engineering(df1)
visualize(df1, df2, OUTPUT_DIR)
test_risk_vs_rat_presence(df1)

print("\n✅ Project pipeline completed successfully!")
