"""
fertilizer_evaluation.py
---------------------------
Loads the saved fertilizer model + test split and exports a full
evaluation report.

Run (after models/fertilizer_model.py has been run at least once):
    python evaluation/fertilizer_evaluation.py
"""
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = os.path.join("models", "saved")
DATA_PATH = os.path.join("data", "fertilizer", "processed", "processed.npz")
RESULTS_DIR = os.path.join("results", "model_results")


def run():
    data = np.load(DATA_PATH, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    classes = data["classes"]

    model = joblib.load(os.path.join(MODEL_DIR, "fertilizer_best_model.joblib"))
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=list(classes),
                                    output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_df.to_csv(os.path.join(RESULTS_DIR, "fertilizer_classification_report.csv"))
    print(f"[fertilizer-eval] Test accuracy: {acc:.4f}")
    print(report_df.round(3))
    print(f"[fertilizer-eval] full per-class report -> {RESULTS_DIR}/fertilizer_classification_report.csv")


if __name__ == "__main__":
    run()
