"""
image_preprocessing.py
------------------------
Preprocessing pipeline for the combined leaf-image dataset (Sugarcane Leaf
Disease Dataset + Cotton Leaf Disease Dataset + PlantVillage Apple-Scab
subset). Images live under data/plant_images/<Crop>_<ClassName>/*.jpg

Steps:
1. Walk the folder tree and build a manifest (filepath, crop, class, label)
2. Stratified train / val / test split (70 / 15 / 15) written to CSV
3. Utility to load images into a resized, normalised numpy array
   (used by models/plant_model.py). A `max_per_class` cap keeps the
   in-memory / CPU-only baseline run fast; the full-resolution / full-count
   path is used automatically when TensorFlow + GPU are available (see
   models/plant_model.py for the Keras ImageDataGenerator version).

Run:
    python preprocessing/image_preprocessing.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_DIR = os.path.join("data", "plant_images")
OUT_DIR = os.path.join("data", "plant_images", "processed")
IMG_SIZE = 64  # resized square dimension used by the CPU baseline CNN/MLP
VALID_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def build_manifest(img_dir=IMG_DIR):
    rows = []
    for class_folder in sorted(os.listdir(img_dir)):
        folder_path = os.path.join(img_dir, class_folder)
        if not os.path.isdir(folder_path) or class_folder == "processed":
            continue
        crop = class_folder.split("_")[0]
        for fname in os.listdir(folder_path):
            if fname.endswith(VALID_EXT):
                rows.append({
                    "filepath": os.path.join(folder_path, fname),
                    "crop": crop,
                    "label": class_folder,
                })
    df = pd.DataFrame(rows)
    print(f"[image] manifest built: {len(df)} images across {df['label'].nunique()} classes")
    return df


def split_manifest(df, save=True):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        train_df.to_csv(os.path.join(OUT_DIR, "train_manifest.csv"), index=False)
        val_df.to_csv(os.path.join(OUT_DIR, "val_manifest.csv"), index=False)
        test_df.to_csv(os.path.join(OUT_DIR, "test_manifest.csv"), index=False)
        print(f"[image] train={len(train_df)} val={len(val_df)} test={len(test_df)} -> {OUT_DIR}")
    return train_df, val_df, test_df


def load_images(df, img_size=IMG_SIZE, max_per_class=None, seed=42):
    """Load & resize images referenced in a manifest dataframe into a
    normalised float32 numpy array (N, img_size, img_size, 3) plus string
    labels. `max_per_class` subsamples for CPU-only fast runs."""
    rng = np.random.RandomState(seed)
    if max_per_class is not None:
        parts = []
        for label, group in df.groupby("label"):
            n = min(len(group), max_per_class)
            parts.append(group.sample(n=n, random_state=seed))
        df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    X, y = [], []
    for _, row in df.iterrows():
        try:
            img = Image.open(row["filepath"]).convert("RGB").resize((img_size, img_size))
            X.append(np.asarray(img, dtype=np.float32) / 255.0)
            y.append(row["label"])
        except Exception as e:
            print(f"[image] skipped {row['filepath']}: {e}")
    return np.stack(X), np.array(y)


if __name__ == "__main__":
    manifest = build_manifest()
    manifest.to_csv(os.path.join(IMG_DIR, "full_manifest.csv"), index=False)
    split_manifest(manifest)
