"""
crop_model.py
--------------
Trains and compares two classifiers for crop recommendation:
  - Decision Tree Classifier
  - Random Forest Classifier
Both are evaluated with accuracy, precision/recall/F1 (macro), and a
confusion matrix. The better model (by macro-F1) is saved for the agent.

Run:
    python models/crop_model.py
"""
import os
import sys
import json
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

sys.path.append(os.getcwd())
from preprocessing.crop_preprocessing import preprocess

MODEL_DIR = os.path.join("models", "saved")
RESULTS_DIR = os.path.join("results", "model_results")
CM_DIR = os.path.join("results", "confusion_matrices")


def evaluate(name, model, X_test, y_test, classes):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
    report = classification_report(y_test, preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=False, cmap="Greens", xticklabels=classes, yticklabels=classes)
    plt.title(f"Crop Recommendation - {name} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(CM_DIR, exist_ok=True)
    plt.savefig(os.path.join(CM_DIR, f"crop_{name.lower().replace(' ', '_')}_cm.png"), dpi=120)
    plt.close()

    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec,
            "f1_macro": f1, "report": report}


def run():
    X_train, X_test, y_train, y_test, scaler, le = preprocess(save=True)
    classes = list(le.classes_)

    results = {}

    # --- Decision Tree ---
    t0 = time.time()
    dt = DecisionTreeClassifier(max_depth=12, random_state=42)
    dt.fit(X_train, y_train)
    dt_time = time.time() - t0
    dt_cv = cross_val_score(dt, X_train, y_train, cv=5).mean()
    dt_res = evaluate("Decision Tree", dt, X_test, y_test, classes)
    dt_res.update({"train_time_sec": dt_time, "cv_accuracy_5fold": dt_cv})
    results["decision_tree"] = dt_res
    print(f"[crop] Decision Tree  -> acc={dt_res['accuracy']:.4f}  f1_macro={dt_res['f1_macro']:.4f}  cv={dt_cv:.4f}")

    # --- Random Forest ---
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_cv = cross_val_score(rf, X_train, y_train, cv=5).mean()
    rf_res = evaluate("Random Forest", rf, X_test, y_test, classes)
    rf_res.update({"train_time_sec": rf_time, "cv_accuracy_5fold": rf_cv})
    results["random_forest"] = rf_res
    print(f"[crop] Random Forest  -> acc={rf_res['accuracy']:.4f}  f1_macro={rf_res['f1_macro']:.4f}  cv={rf_cv:.4f}")

    # --- Feature importance (RF) ---
    importances = rf.feature_importances_
    feat_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    plt.figure(figsize=(7, 5))
    order = np.argsort(importances)[::-1]
    plt.bar([feat_names[i] for i in order], importances[order], color="#3f7d3f")
    plt.title("Random Forest - Feature Importance (Crop Recommendation)")
    plt.ylabel("Importance")
    plt.tight_layout()
    os.makedirs(os.path.join("results", "graphs"), exist_ok=True)
    plt.savefig(os.path.join("results", "graphs", "crop_rf_feature_importance.png"), dpi=120)
    plt.close()

    # --- Select best model by macro F1 ---
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = {"decision_tree": dt, "random_forest": rf}[best_name]
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "crop_best_model.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "crop_label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "crop_scaler.joblib"))
    print(f"[crop] BEST MODEL = {best_name} -> saved to {MODEL_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "report"} for k, v in results.items()}
    summary["best_model"] = best_name
    with open(os.path.join(RESULTS_DIR, "crop_model_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[crop] metrics saved -> {RESULTS_DIR}/crop_model_results.json")

    return results


if __name__ == "__main__":
    run()
