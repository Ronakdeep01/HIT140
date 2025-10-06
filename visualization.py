import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid", palette="Set2")

def visualize(df1, df2, output_dir):
    plt.figure(figsize=(8,5))
    sns.histplot(df1['hours_after_sunset'], kde=True)
    plt.title('Bat Landings – Hours After Sunset')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bat_landings_hours_after_sunset.png'))
    plt.close()

    plt.figure(figsize=(7,5))
    sns.countplot(data=df1, x='season', hue='risk')
    plt.title('Risk-Taking Behavior by Season')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'risk_by_season.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    sns.heatmap(df2.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap – Dataset 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset2_heatmap.png'))
    plt.close()

    print("✅ Visualizations saved!")

if __name__ == "__main__":
    print("Visualization module loaded successfully.")
