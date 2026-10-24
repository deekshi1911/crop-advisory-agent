"""
plant_model.py
----------------
Plant leaf-disease classifier over the combined image dataset
(Sugarcane Leaf Disease Dataset + Cotton Leaf Disease Dataset +
PlantVillage Apple-Scab subset), 10 classes total.

Two code paths are provided:

1. build_cnn() / train_keras_cnn()
   A real Convolutional Neural Network (Conv2D + MaxPooling stacks,
   BatchNorm, Dropout, Dense softmax head) built with tf.keras and trained
   with ImageDataGenerator augmentation. This is the path the project
   should use for the final report -- run it on Google Colab (free GPU)
   or any machine with TensorFlow installed; this repo's requirements.txt
   lists tensorflow for that purpose.

2. train_cpu_baseline()
   A lightweight CPU-only baseline (HOG image features -> MLPClassifier)
   that runs anywhere, with no TensorFlow / GPU required. It is used
   automatically when TensorFlow is not importable, which is the case in
   this development sandbox (no internet access to install TensorFlow).
   It lets the pipeline be fully demonstrated end-to-end (train, validate,
   test, confusion matrix, accuracy) even without GPU/TF installed.

Run:
    python models/plant_model.py
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
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.getcwd())
from preprocessing.image_preprocessing import (build_manifest, split_manifest,
                                                load_images, IMG_SIZE)

MODEL_DIR = os.path.join("models", "saved")
RESULTS_DIR = os.path.join("results", "model_results")
CM_DIR = os.path.join("results", "confusion_matrices")
GRAPH_DIR = os.path.join("results", "graphs")

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------
# Path 1: real CNN (Keras) -- use on GPU / Colab
# ---------------------------------------------------------------------
def build_cnn(input_shape=(128, 128, 3), num_classes=10):
    """A compact CNN: 4 Conv blocks -> GAP -> Dense softmax."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1. / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_keras_cnn(img_dir=os.path.join("data", "plant_images"),
                     img_size=128, batch_size=32, epochs=25):
    """Train the real CNN with on-the-fly augmentation. Requires
    TensorFlow (`pip install tensorflow`) and is best run with a GPU."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rescale=1. / 255, validation_split=0.15,
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.15, horizontal_flip=True,
    )
    train_gen = datagen.flow_from_directory(
        img_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode="sparse", subset="training", seed=42)
    val_gen = datagen.flow_from_directory(
        img_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode="sparse", subset="validation", seed=42)

    model = build_cnn(input_shape=(img_size, img_size, 3),
                       num_classes=train_gen.num_classes)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs,
                         callbacks=[early_stop])

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "plant_cnn_model.keras"))

    # accuracy / loss curves
    os.makedirs(GRAPH_DIR, exist_ok=True)
    for metric in ["accuracy", "loss"]:
        plt.figure()
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"CNN {metric} over epochs")
        plt.xlabel("epoch"); plt.legend()
        plt.savefig(os.path.join(GRAPH_DIR, f"plant_cnn_{metric}_curve.png"), dpi=120)
        plt.close()

    return model, train_gen, val_gen, history


# ---------------------------------------------------------------------
# Path 2: CPU-only baseline (HOG features + MLP) -- runs without TF/GPU
# ---------------------------------------------------------------------
def extract_hog_features(X):
    """X: (N, H, W, 3) float32 in [0,1] -> HOG feature matrix."""
    from skimage.feature import hog
    from skimage.color import rgb2gray
    feats = []
    for img in X:
        gray = rgb2gray(img)
        f = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), feature_vector=True)
        feats.append(f)
    return np.array(feats)


def train_cpu_baseline(max_per_class=140, img_size=IMG_SIZE):
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier

    print("[plant] TensorFlow not available in this environment -> "
          "running CPU-only HOG+ML baseline (see build_cnn()/train_keras_cnn() "
          "for the real CNN to run on GPU/Colab).")

    manifest = build_manifest()
    manifest.to_csv(os.path.join("data", "plant_images", "full_manifest.csv"), index=False)
    train_df, val_df, test_df = split_manifest(manifest)

    t0 = time.time()
    X_train_img, y_train_lbl = load_images(train_df, img_size=img_size, max_per_class=max_per_class)
    X_val_img, y_val_lbl = load_images(val_df, img_size=img_size, max_per_class=max(1, max_per_class // 5))
    X_test_img, y_test_lbl = load_images(test_df, img_size=img_size, max_per_class=max(1, max_per_class // 5))
    print(f"[plant] loaded images in {time.time()-t0:.1f}s: "
          f"train={len(X_train_img)} val={len(X_val_img)} test={len(X_test_img)}")

    le = LabelEncoder()
    le.fit(manifest["label"])
    y_train = le.transform(y_train_lbl)
    y_val = le.transform(y_val_lbl)
    y_test = le.transform(y_test_lbl)
    classes = list(le.classes_)

    t0 = time.time()
    X_train_feat = extract_hog_features(X_train_img)
    X_val_feat = extract_hog_features(X_val_img)
    X_test_feat = extract_hog_features(X_test_img)
    print(f"[plant] HOG feature extraction in {time.time()-t0:.1f}s, dim={X_train_feat.shape[1]}")

    results = {}

    # --- MLP (fully-connected NN, CPU-friendly stand-in for the CNN head) ---
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                         early_stopping=True, random_state=42)
    mlp.fit(X_train_feat, y_train)
    mlp_time = time.time() - t0
    val_acc = accuracy_score(y_val, mlp.predict(X_val_feat))
    results["mlp_baseline"] = _eval_and_plot("MLP (HOG features)", mlp, X_test_feat, y_test,
                                              classes, "plant_mlp")
    results["mlp_baseline"].update({"train_time_sec": mlp_time, "val_accuracy": val_acc})
    print(f"[plant] MLP baseline -> val_acc={val_acc:.4f} test_acc={results['mlp_baseline']['accuracy']:.4f}")

    # --- Random Forest on HOG features (comparison) ---
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train_feat, y_train)
    rf_time = time.time() - t0
    val_acc_rf = accuracy_score(y_val, rf.predict(X_val_feat))
    results["rf_baseline"] = _eval_and_plot("Random Forest (HOG features)", rf, X_test_feat, y_test,
                                             classes, "plant_rf")
    results["rf_baseline"].update({"train_time_sec": rf_time, "val_accuracy": val_acc_rf})
    print(f"[plant] RF baseline  -> val_acc={val_acc_rf:.4f} test_acc={results['rf_baseline']['accuracy']:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = {"mlp_baseline": mlp, "rf_baseline": rf}[best_name]
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "plant_best_baseline_model.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "plant_label_encoder.joblib"))
    print(f"[plant] BEST BASELINE MODEL = {best_name} -> saved to {MODEL_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "report"} for k, v in results.items()}
    summary["best_model"] = best_name
    summary["note"] = ("CPU-only HOG+ML baseline trained in the dev sandbox "
                        "(no TensorFlow/GPU access). See build_cnn()/train_keras_cnn() "
                        "in this file for the real Conv2D CNN to run on Colab/GPU for "
                        "the final submission.")
    with open(os.path.join(RESULTS_DIR, "plant_model_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[plant] metrics saved -> {RESULTS_DIR}/plant_model_results.json")

    return results


def _eval_and_plot(name, model, X_test, y_test, classes, tag):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    report = classification_report(y_test, preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Plant Disease - {name} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=75, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(CM_DIR, exist_ok=True)
    plt.savefig(os.path.join(CM_DIR, f"{tag}_cm.png"), dpi=120)
    plt.close()

    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec,
            "f1_macro": f1, "report": report}


if __name__ == "__main__":
    if TF_AVAILABLE:
        train_keras_cnn()
    else:
        train_cpu_baseline()
