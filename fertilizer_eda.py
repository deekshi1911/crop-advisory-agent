"""
fertilizer_eda.py
-------------------
Exploratory Data Analysis for the Fertilizer Recommendation dataset.
Generates and saves plots to results/graphs/fertilizer_*.png

Run:
    python eda/fertilizer_eda.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RAW_PATH = os.path.join("data", "fertilizer", "fertilizer_recommendation_dataset.csv")
OUT_DIR = os.path.join("results", "graphs")
NUMERIC = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]

    print(df.describe())
    print("\nFertilizer class balance:\n", df["Fertilizer Name"].value_counts())
    print("\nSoil Type balance:\n", df["Soil Type"].value_counts())
    print("\nCrop Type balance:\n", df["Crop Type"].value_counts())

    # 1. Target class distribution
    plt.figure(figsize=(10, 5))
    df["Fertilizer Name"].value_counts().plot(kind="bar", color="#a0522d")
    plt.title("Fertilizer Recommendation - Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fertilizer_class_distribution.png"), dpi=120)
    plt.close()

    # 2. Soil Type & Crop Type distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["Soil Type"].value_counts().plot(kind="bar", ax=axes[0], color="#c2a878")
    axes[0].set_title("Soil Type distribution")
    df["Crop Type"].value_counts().plot(kind="bar", ax=axes[1], color="#6b8e23")
    axes[1].set_title("Crop Type distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fertilizer_categorical_distribution.png"), dpi=120)
    plt.close()

    # 3. Correlation heatmap (numeric)
    plt.figure(figsize=(7, 6))
    sns.heatmap(df[NUMERIC].corr(), annot=True, cmap="OrRd", fmt=".2f")
    plt.title("Fertilizer Dataset - Feature Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fertilizer_correlation_heatmap.png"), dpi=120)
    plt.close()

    # 4. Numeric feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, NUMERIC):
        sns.histplot(df[col], kde=True, ax=ax, color="#4682b4")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fertilizer_feature_distributions.png"), dpi=120)
    plt.close()

    print(f"[fertilizer] EDA plots saved to {OUT_DIR}")


if __name__ == "__main__":
    run()
