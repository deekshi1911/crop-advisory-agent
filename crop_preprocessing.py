"""
crop_preprocessing.py
----------------------
Preprocessing pipeline for the Crop Recommendation dataset.

Steps performed:
1. Load raw CSV (N, P, K, temperature, humidity, ph, rainfall, label)
2. Handle missing values / duplicates
3. Encode the target label (crop name) with LabelEncoder
4. Detect and report outliers (IQR method) -- reported, not removed, since
   agronomic values naturally vary a lot across crops
5. Scale numeric features with StandardScaler
6. Train / test split (80/20, stratified on label)
7. Persist processed arrays + fitted encoders/scalers to disk (joblib)

Run:
    python preprocessing/crop_preprocessing.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RAW_PATH = os.path.join("data", "crop", "crop_recommendation_dataset.csv")
OUT_DIR = os.path.join("data", "crop", "processed")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df):
    before = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"[crop] dropped {before - len(df)} rows (nulls/duplicates); {len(df)} remain")
    return df


def report_outliers(df, cols=FEATURES):
    report = {}
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[c] < lo) | (df[c] > hi)).sum()
        report[c] = int(n_out)
    print("[crop] outlier counts (IQR method):", report)
    return report


def preprocess(save=True):
    df = load_raw()
    df = clean(df)
    report_outliers(df)

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    X = df[FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
        joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))
        np.savez(
            os.path.join(OUT_DIR, "processed.npz"),
            X_train=X_train_scaled, X_test=X_test_scaled,
            X_train_raw=X_train.values, X_test_raw=X_test.values,
            y_train=y_train, y_test=y_test,
            feature_names=np.array(FEATURES), classes=le.classes_,
        )
        print(f"[crop] saved processed data -> {OUT_DIR}/processed.npz")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


if __name__ == "__main__":
    preprocess()
