"""
crop_eda.py
------------
Exploratory Data Analysis for the Crop Recommendation dataset.
Generates and saves plots to results/graphs/crop_*.png

Run:
    python eda/crop_eda.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RAW_PATH = os.path.join("data", "crop", "crop_recommendation_dataset.csv")
OUT_DIR = os.path.join("results", "graphs")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]

    print(df.describe())
    print("\nClass balance:\n", df["label"].value_counts())

    # 1. Class distribution
    plt.figure(figsize=(12, 5))
    df["label"].value_counts().plot(kind="bar", color="#3f7d3f")
    plt.title("Crop Recommendation - Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "crop_class_distribution.png"), dpi=120)
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Crop Dataset - Feature Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "crop_correlation_heatmap.png"), dpi=120)
    plt.close()

    # 3. Feature distributions
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#2e6f95")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "crop_feature_distributions.png"), dpi=120)
    plt.close()

    # 4. Boxplots per feature (outlier visualisation)
    plt.figure(figsize=(14, 6))
    df_melt = df[FEATURES].melt(var_name="feature", value_name="value")
    sns.boxplot(data=df_melt, x="feature", y="value")
    plt.title("Crop Dataset - Boxplots (outlier check)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "crop_boxplots.png"), dpi=120)
    plt.close()

    print(f"[crop] EDA plots saved to {OUT_DIR}")


if __name__ == "__main__":
    run()
