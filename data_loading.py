import pandas as pd
import os

PATH_DATASET1 = r"D:\dataset1 (2).csv"
PATH_DATASET2 = r"D:\dataset2 (1).csv"
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_datasets():
    try:
        df1 = pd.read_csv(PATH_DATASET1)
        df2 = pd.read_csv(PATH_DATASET2)
        print("✅ Data loaded successfully")
        print("Dataset1 shape:", df1.shape)
        print("Dataset2 shape:", df2.shape)
        return df1, df2
    except Exception as e:
        raise SystemExit(f"Error loading datasets: {e}")

if __name__ == "__main__":
    df1, df2 = load_datasets()
