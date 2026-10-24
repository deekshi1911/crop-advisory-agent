"""
fertilizer_preprocessing.py
-----------------------------
Preprocessing pipeline for the Fertilizer Recommendation dataset.

Columns: Temperature, Humidity, Moisture, Soil Type, Crop Type,
         Nitrogen, Potassium, Phosphorous, Fertilizer Name (target)

Steps:
1. Load raw CSV, clean column names, drop nulls/duplicates
2. Label-encode the two categorical predictors (Soil Type, Crop Type)
3. Label-encode the target (Fertilizer Name)
4. Scale numeric features with StandardScaler
5. Stratified train/test split (80/20)
6. Persist processed arrays + encoders/scaler

Run:
    python preprocessing/fertilizer_preprocessing.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RAW_PATH = os.path.join("data", "fertilizer", "fertilizer_recommendation_dataset.csv")
OUT_DIR = os.path.join("data", "fertilizer", "processed")
NUMERIC = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
CATEGORICAL = ["Soil Type", "Crop Type"]
TARGET = "Fertilizer Name"


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df):
    before = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"[fertilizer] dropped {before - len(df)} rows (nulls/duplicates); {len(df)} remain")
    return df


def preprocess(save=True):
    df = load_raw()
    df = clean(df)

    cat_encoders = {}
    for col in CATEGORICAL:
        enc = LabelEncoder()
        df[col + "_enc"] = enc.fit_transform(df[col])
        cat_encoders[col] = enc

    target_enc = LabelEncoder()
    y = target_enc.fit_transform(df[TARGET])

    feature_cols = NUMERIC + [c + "_enc" for c in CATEGORICAL]
    X = df[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
        joblib.dump(cat_encoders, os.path.join(OUT_DIR, "cat_encoders.joblib"))
        joblib.dump(target_enc, os.path.join(OUT_DIR, "target_encoder.joblib"))
        np.savez(
            os.path.join(OUT_DIR, "processed.npz"),
            X_train=X_train_scaled, X_test=X_test_scaled,
            y_train=y_train, y_test=y_test,
            feature_names=np.array(feature_cols), classes=target_enc.classes_,
        )
        print(f"[fertilizer] saved processed data -> {OUT_DIR}/processed.npz")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, target_enc


if __name__ == "__main__":
    preprocess()
