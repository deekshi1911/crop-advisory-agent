"""
crop_advisory_agent.py
-------------------------
Ties the trained models together into one advisory agent:

  1. Crop Recommendation       (N, P, K, temperature, humidity, ph, rainfall)
  2. Fertilizer Recommendation (Temperature, Humidity, Moisture, Soil Type,
                                Crop Type, N, K, P)
  3. Plant Disease Detection   (leaf image) -> disease class + cause/treatment
                                looked up in data/disease_recommendations.csv

Changes in this version
  * all paths are absolute (works no matter which folder you launch from)
  * top-N predictions with probabilities (not just the single best answer)
  * FIX: the crop model knows 45 crops but the fertilizer model only knows
    18 crop types, so the combined advisory crashed for 33 crops. The
    recommended crop is now mapped to the nearest fertilizer crop type.
  * disease detection returns top-3 + a low-confidence flag
  * OPTIONAL: if models/saved/crop_disease_cnn.keras exists (the CNN from your
    notebook) and TensorFlow is installed, it is used for disease detection;
    otherwise the HOG + Random Forest baseline is used.

Usage:
    python agent/crop_advisory_agent.py
(also importable: from agent.crop_advisory_agent import CropAdvisoryAgent)
"""
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
from preprocessing.image_preprocessing import IMG_SIZE  # noqa: E402

MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
DISEASE_CSV = os.path.join(BASE_DIR, "data", "disease_recommendations.csv")
CAT_ENCODERS = os.path.join(BASE_DIR, "data", "fertilizer", "processed", "cat_encoders.joblib")
FERT_CSV = os.path.join(BASE_DIR, "data", "fertilizer", "fertilizer_recommendation_dataset.csv")

# Crop-recommendation label -> crop type known to the fertilizer model.
# Entries marked approximate=True are the closest available type, not an exact match.
CROP_TO_FERT_TYPE = {
    # exact / near-exact
    "barley": ("Barley", False), "coffee": ("Coffee", False), "cotton": ("Cotton", False),
    "maize": ("Maize", False), "mustard": ("Mustard", False), "rice": ("Rice", False),
    "soybean": ("Soybean", False), "sugarcane": ("Sugarcane", False),
    "sunflower": ("Sunflower", False), "tea": ("Tea", False),
    "tobacco": ("Tobacco", False), "wheat": ("Wheat", False),
    "groundnut": ("Ground Nuts", False),
    # grouped
    "bajra": ("Millets", False), "jowar": ("Millets", False), "ragi": ("Millets", False),
    "sesame": ("Oil seeds", False), "castor": ("Oil seeds", False),
    "chickpea": ("Pulses", False), "lentil": ("Pulses", False), "blackgram": ("Pulses", False),
    "mungbean": ("Pulses", False), "mothbeans": ("Pulses", False),
    "pigeonpea": ("Pulses", False), "kidneybeans": ("Pulses", False),
    "apple": ("Fruits", False), "banana": ("Fruits", False), "mango": ("Fruits", False),
    "orange": ("Fruits", False), "papaya": ("Fruits", False), "grapes": ("Fruits", False),
    "pomegranate": ("Fruits", False), "muskmelon": ("Fruits", False),
    "watermelon": ("Fruits", False), "coconut": ("Fruits", False),
    "tomato": ("Vegetables", False), "potato": ("Vegetables", False),
    "onion": ("Vegetables", False), "chilli": ("Vegetables", False),
    # approximate
    "ginger": ("Vegetables", True), "turmeric": ("Vegetables", True),
    "blackpepper": ("Vegetables", True), "cardamom": ("Vegetables", True),
    "jute": ("Cotton", True), "rubber": ("Tea", True),
}

# Notebook CNN class -> (crop, disease label used in disease_recommendations.csv)
CNN_CLASSES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow",
               "bacterial_blight", "curl_virus", "fusarium_wilt"]
CNN_CROP = {"Healthy": "Sugarcane/Cotton", "Mosaic": "Sugarcane", "RedRot": "Sugarcane",
            "Rust": "Sugarcane", "Yellow": "Sugarcane", "bacterial_blight": "Cotton",
            "curl_virus": "Cotton", "fusarium_wilt": "Cotton"}
CNN_ALIAS = {"fusarium_wilt": "fussarium_wilt"}      # spelling used in the CSV

LOW_CONFIDENCE = 0.50

try:                                        # shown when a model cannot be loaded cleanly
    import sklearn
    _SK = sklearn.__version__
except Exception:                           # pragma: no cover
    _SK = "unknown"
FIX_HINT = (f"Installed scikit-learn is {_SK}, the saved models were built with 1.8.0. "
            "Fix: run  pip install scikit-learn==1.8.0  (needs Python 3.11+), "
            "or rebuild the models with  python retrain_models.py")


def _norm(s):
    return str(s).lower().replace(" ", "").replace("_", "")


class CropAdvisoryAgent:
    def __init__(self):
        self.model_issues = []
        self.crop_model = self._safe_load("crop_best_model.joblib")
        self.crop_scaler = self._safe_load("crop_scaler.joblib")
        self.crop_le = self._safe_load("crop_label_encoder.joblib")

        self.fert_model = self._safe_load("fertilizer_best_model.joblib")
        self.fert_scaler = self._safe_load("fertilizer_scaler.joblib")
        self.fert_le = self._safe_load("fertilizer_label_encoder.joblib")
        self.cat_encoders = joblib.load(CAT_ENCODERS) if os.path.exists(CAT_ENCODERS) else None

        self.plant_model = self._safe_load("plant_best_baseline_model.joblib")
        self.plant_le = self._safe_load("plant_label_encoder.joblib")
        self.cnn = self._load_cnn()

        self.disease_df = pd.read_csv(DISEASE_CSV) if os.path.exists(DISEASE_CSV) else None

        # training ranges of the fertilizer dataset, used to warn about out-of-range inputs
        self.fert_ranges = {}
        if os.path.exists(FERT_CSV):
            f = pd.read_csv(FERT_CSV)
            f.columns = f.columns.str.strip()
            for col in ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]:
                self.fert_ranges[col] = (float(f[col].min()), float(f[col].max()))

    # ------------------------------------------------------------ loading
    def _safe_load(self, fname):
        """Load a .joblib file. A scikit-learn version mismatch or a corrupt file is
        recorded in self.model_issues (shown in the web UI) instead of crashing."""
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            return None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                obj = joblib.load(path)
            except Exception as exc:
                self.model_issues.append(f"{fname} could not be loaded ({type(exc).__name__}). {FIX_HINT}")
                return None
        if any("version" in str(w.message).lower() and "sklearn" in str(w.category).lower() + str(w.message).lower()
               for w in caught):
            msg = f"{fname} was saved with a different scikit-learn version. {FIX_HINT}"
            if msg not in self.model_issues:
                self.model_issues.append(msg)
        return obj

    @staticmethod
    def _load_cnn():
        path = os.path.join(MODEL_DIR, "crop_disease_cnn.keras")
        if not os.path.exists(path):
            return None
        try:
            from tensorflow.keras.models import load_model
            return load_model(path)
        except Exception as exc:                      # TF missing / bad file
            print(f"[agent] CNN not loaded ({exc}); using the HOG baseline.")
            return None

    @property
    def disease_backend(self):
        return "CNN (Keras)" if self.cnn is not None else "HOG + Random Forest baseline"

    def _unavailable(self, name, script):
        if self.model_issues:
            return (f"The {name} model is not available because a model file failed to load. "
                    f"{FIX_HINT}")
        return f"The {name} model is not trained yet - run {script}"

    # ----------------------------------------------------- crop recommendation
    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall, top_k=5):
        if None in (self.crop_model, self.crop_scaler, self.crop_le):
            return {"error": self._unavailable("crop", "models/crop_model.py")}
        X = self.crop_scaler.transform(np.array([[N, P, K, temperature, humidity, ph, rainfall]]))
        proba = self.crop_model.predict_proba(X)[0]
        order = np.argsort(proba)[::-1][:top_k]
        classes = self.crop_le.inverse_transform(self.crop_model.classes_)
        top = [{"name": str(classes[i]), "probability": float(proba[i])} for i in order]
        return {"recommended_crop": top[0]["name"], "confidence": top[0]["probability"], "top": top}

    # ------------------------------------------------- fertilizer recommendation
    def recommend_fertilizer(self, temperature, humidity, moisture,
                             soil_type, crop_type, nitrogen, potassium, phosphorous, top_k=3):
        if None in (self.fert_model, self.fert_scaler, self.fert_le, self.cat_encoders):
            return {"error": self._unavailable("fertilizer", "models/fertilizer_model.py")}
        soil_enc = self.cat_encoders["Soil Type"].transform([soil_type])[0]
        crop_enc = self.cat_encoders["Crop Type"].transform([crop_type])[0]
        X = np.array([[temperature, humidity, moisture, nitrogen, potassium, phosphorous,
                       soil_enc, crop_enc]])
        X = self.fert_scaler.transform(X)
        proba = self.fert_model.predict_proba(X)[0]
        order = np.argsort(proba)[::-1][:top_k]
        classes = self.fert_le.inverse_transform(self.fert_model.classes_)
        top = [{"name": str(classes[i]), "probability": float(proba[i])} for i in order]
        return {"recommended_fertilizer": top[0]["name"], "confidence": top[0]["probability"],
                "top": top, "crop_type_used": crop_type, "soil_type_used": soil_type}

    def fertilizer_warnings(self, temperature, humidity, moisture, nitrogen, potassium, phosphorous):
        """List inputs that fall outside the range the fertilizer model was trained on."""
        vals = {"Temperature": temperature, "Humidity": humidity, "Moisture": moisture,
                "Nitrogen": nitrogen, "Potassium": potassium, "Phosphorous": phosphorous}
        out = []
        for name, v in vals.items():
            lo, hi = self.fert_ranges.get(name, (None, None))
            if lo is not None and v is not None and not (lo <= v <= hi):
                out.append(f"{name} = {v:g} is outside the fertilizer model's training range "
                           f"({lo:g}-{hi:g}); treat this suggestion with extra caution.")
        return out

    def fertilizer_crop_type(self, crop_name):
        """Map a crop-model label to a crop type the fertilizer model understands."""
        if self.cat_encoders is None:
            return (str(crop_name).capitalize(), True)
        known = list(self.cat_encoders["Crop Type"].classes_)
        hit = CROP_TO_FERT_TYPE.get(str(crop_name).lower())
        if hit and hit[0] in known:
            return hit
        return (known[0], True)

    # -------------------------------------------------- disease lookup / detection
    def _lookup_advisory(self, crop, disease):
        if self.disease_df is None:
            return None
        df = self.disease_df
        crops = [c.strip().lower() for c in str(crop).split("/")]
        m = df[df["crop"].str.lower().isin(crops) &
               (df["disease_label"].map(_norm) == _norm(CNN_ALIAS.get(disease, disease)))]
        return m.iloc[0].to_dict() if not m.empty else None

    def detect_disease(self, image_path):
        if self.cnn is not None:
            return self._detect_cnn(image_path)
        if self.plant_model is None or self.plant_le is None:
            return {"error": self._unavailable("disease", "models/plant_model.py")}
        from preprocessing.plant_model_utils import single_image_features
        feats = single_image_features(image_path, img_size=IMG_SIZE)
        proba = self.plant_model.predict_proba([feats])[0]
        order = np.argsort(proba)[::-1][:3]
        labels = self.plant_le.inverse_transform(self.plant_model.classes_)
        top = [{"label": str(labels[i]), "probability": float(proba[i])} for i in order]
        label = top[0]["label"]
        crop, disease = label.split("_", 1)
        return self._disease_result(crop, disease, label, top, top[0]["probability"])

    def _detect_cnn(self, image_path):
        from PIL import Image
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.asarray(img, dtype=np.float32) / 255.0, 0)
        proba = self.cnn.predict(arr, verbose=0)[0]
        order = np.argsort(proba)[::-1][:3]
        top = [{"label": CNN_CLASSES[i], "probability": float(proba[i])} for i in order]
        disease = top[0]["label"]
        crop = CNN_CROP[disease]
        return self._disease_result(crop, disease, f"{crop}_{disease}", top, top[0]["probability"])

    def _disease_result(self, crop, disease, label, top, conf):
        return {
            "crop": crop, "disease": disease.replace("_", " "),
            "predicted_label": label, "confidence": conf, "top": top,
            "is_healthy": disease.lower() == "healthy",
            "low_confidence": conf < LOW_CONFIDENCE,
            "backend": self.disease_backend,
            "advisory": self._lookup_advisory(crop, disease),
        }

    # ----------------------------------------------------- combined advisory
    def full_advisory(self, soil_conditions, image_path=None, fertilizer_crop_type=None):
        """soil_conditions: dict with N,P,K,temperature,humidity,ph,rainfall,
        moisture, soil_type.  fertilizer_crop_type: optional manual override."""
        s = soil_conditions
        crop_result = self.recommend_crop(s["N"], s["P"], s["K"], s["temperature"],
                                          s["humidity"], s["ph"], s["rainfall"])
        if "error" in crop_result:
            return {"crop_recommendation": crop_result}

        crop_name = crop_result["recommended_crop"]
        if fertilizer_crop_type:
            fert_type, approx = fertilizer_crop_type, False
        else:
            fert_type, approx = self.fertilizer_crop_type(crop_name)

        fert_result = self.recommend_fertilizer(
            s["temperature"], s["humidity"], s.get("moisture", 30), s.get("soil_type", "Loamy"),
            fert_type, s["N"], s["K"], s["P"])
        if "error" not in fert_result:
            fert_result["approximate_crop_match"] = approx
            fert_result["warnings"] = self.fertilizer_warnings(
                s["temperature"], s["humidity"], s.get("moisture", 30), s["N"], s["K"], s["P"])

        result = {"crop_recommendation": crop_result, "fertilizer_recommendation": fert_result}
        if image_path:
            result["disease_detection"] = self.detect_disease(image_path)
        return result


if __name__ == "__main__":
    import json
    agent = CropAdvisoryAgent()
    demo = {"N": 90, "P": 42, "K": 43, "temperature": 25.0, "humidity": 80.0,
            "ph": 6.5, "rainfall": 200.0, "moisture": 35, "soil_type": "Loamy"}
    print(json.dumps(agent.full_advisory(demo), indent=2, default=str))
