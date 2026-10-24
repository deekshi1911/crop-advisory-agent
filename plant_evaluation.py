"""
plant_evaluation.py
----------------------
Prints/exports the saved plant-disease baseline model results
(results/model_results/plant_model_results.json produced by
models/plant_model.py) as a tidy comparison table.

Run:
    python evaluation/plant_evaluation.py
"""
import os
import json
import pandas as pd

RESULTS_DIR = os.path.join("results", "model_results")


def run():
    path = os.path.join(RESULTS_DIR, "plant_model_results.json")
    with open(path) as f:
        results = json.load(f)

    rows = []
    for model_name, metrics in results.items():
        if model_name in ("best_model", "note"):
            continue
        rows.append({
            "model": model_name,
            "accuracy": round(metrics["accuracy"], 4),
            "val_accuracy": round(metrics.get("val_accuracy", float("nan")), 4),
            "precision_macro": round(metrics["precision_macro"], 4),
            "recall_macro": round(metrics["recall_macro"], 4),
            "f1_macro": round(metrics["f1_macro"], 4),
            "train_time_sec": round(metrics["train_time_sec"], 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "plant_model_comparison.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\nBest model: {results['best_model']}")
    print(results.get("note", ""))


if __name__ == "__main__":
    run()
