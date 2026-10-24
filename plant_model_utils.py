"""
plant_model_utils.py
-----------------------
Small helper used by agent/crop_advisory_agent.py to turn a single leaf
image into the same HOG feature vector the CPU baseline model (in
models/plant_model.py) was trained on.
"""
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray


def single_image_features(image_path, img_size=64):
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = rgb2gray(arr)
    feats = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), feature_vector=True)
    return feats
