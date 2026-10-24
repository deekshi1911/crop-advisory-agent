"""
image_eda.py
-------------
Exploratory Data Analysis for the combined leaf-image dataset.
Generates: class distribution bar chart, sample image grid per class,
image dimension / aspect-ratio summary, mean-brightness per class.

Run:
    python eda/image_eda.py
"""
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMG_DIR = os.path.join("data", "plant_images")
OUT_DIR = os.path.join("results", "graphs")
VALID_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def collect_classes(img_dir=IMG_DIR):
    classes = {}
    for folder in sorted(os.listdir(img_dir)):
        path = os.path.join(img_dir, folder)
        if os.path.isdir(path) and folder != "processed":
            files = [f for f in os.listdir(path) if f.endswith(VALID_EXT)]
            classes[folder] = files
    return classes


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    classes = collect_classes()

    counts = {k: len(v) for k, v in classes.items()}
    print("[image] class counts:", counts)

    # 1. Class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(counts.keys(), counts.values(), color="#556b2f")
    plt.xticks(rotation=75, ha="right")
    plt.title("Leaf Image Dataset - Class Distribution")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "image_class_distribution.png"), dpi=120)
    plt.close()

    # 2. Sample grid (one row per class, 4 samples)
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, 4, figsize=(10, 2.2 * n_classes))
    random.seed(42)
    for row, (cls, files) in enumerate(classes.items()):
        sample = random.sample(files, min(4, len(files)))
        for col in range(4):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(sample):
                img = Image.open(os.path.join(IMG_DIR, cls, sample[col])).convert("RGB")
                ax.imshow(img)
            if col == 0:
                ax.set_ylabel(cls, fontsize=8)
                ax.text(-0.1, 0.5, cls, fontsize=7, rotation=90,
                        va="center", ha="center", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "image_sample_grid.png"), dpi=110)
    plt.close()

    # 3. Image size / brightness summary (sampled for speed)
    widths, heights, brightness = [], [], []
    random.seed(0)
    for cls, files in classes.items():
        for f in random.sample(files, min(20, len(files))):
            img = Image.open(os.path.join(IMG_DIR, cls, f)).convert("RGB")
            w, h = img.size
            widths.append(w); heights.append(h)
            brightness.append(np.asarray(img).mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(widths, heights, alpha=0.4, s=10, color="#8b4513")
    axes[0].set_xlabel("Width (px)"); axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Sampled image dimensions")
    axes[1].hist(brightness, bins=30, color="#daa520")
    axes[1].set_title("Sampled mean pixel brightness")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "image_size_brightness.png"), dpi=120)
    plt.close()

    print(f"[image] EDA plots saved to {OUT_DIR}")


if __name__ == "__main__":
    run()
