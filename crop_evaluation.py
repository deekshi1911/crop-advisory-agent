"""
crop_evaluation.py
--------------------
Loads the saved crop model + test split and prints/exports a full
evaluation report (accuracy, macro/weighted P/R/F1, per-class report).

Run (after models/crop_model.py has been run at least once):
    python evaluation/crop_evaluation.py
"""
import os
import json
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = os.path.join("models", "saved")
DATA_PATH = os.path.join("data", "crop", "processed", "processed.npz")
RESULTS_DIR = os.path.join("results", "model_results")


def run():
    data = np.load(DATA_PATH, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    classes = data["classes"]

    model = joblib.load(os.path.join(MODEL_DIR, "crop_best_model.joblib"))
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=list(classes), output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_df.to_csv(os.path.join(RESULTS_DIR, "crop_classification_report.csv"))
    print(f"[crop-eval] Test accuracy: {acc:.4f}")
    print(report_df.round(3))
    print(f"[crop-eval] full per-class report -> {RESULTS_DIR}/crop_classification_report.csv")


if __name__ == "__main__":
    run()
