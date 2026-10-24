"""
app.py
-------
Crop Advisory Agent - Flask web application.

  /             home / overview
  /crop         crop recommendation        (N, P, K, temp, humidity, pH, rainfall)
  /fertilizer   fertilizer recommendation  (soil + crop + nutrients)
  /disease      leaf-image upload -> disease detection + treatment advice
  /advisory     combined report: crop + fertilizer (+ optional disease)

  JSON API:  POST /api/crop  /api/fertilizer  /api/recommend    (JSON body)
             POST /api/disease                                   (multipart: image)
             GET  /api/health

Run (from anywhere):
    pip install -r requirements.txt
    python app/app.py
Then open http://127.0.0.1:5000
"""
import os
import sys
import uuid
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, render_template, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from agent.crop_advisory_agent import CropAdvisoryAgent

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UPLOAD_MAX_AGE = 24 * 3600          # uploaded photos are deleted after 24 h

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024        # 8 MB upload cap

agent = CropAdvisoryAgent()
SOIL_TYPES = list(agent.cat_encoders["Soil Type"].classes_)
CROP_TYPES = list(agent.cat_encoders["Crop Type"].classes_)

# (label, min, max, step, default) - limits follow the training data with some headroom
CROP_FIELDS = {
    "N":           ("Nitrogen - N (kg/ha)",     0, 200, 1,    90),
    "P":           ("Phosphorous - P (kg/ha)",  0, 200, 1,    42),
    "K":           ("Potassium - K (kg/ha)",    0, 250, 1,    43),
    "temperature": ("Temperature (°C)",        -5,  50, 0.1,  25),
    "humidity":    ("Humidity (%)",             0, 100, 0.1,  80),
    "ph":          ("Soil pH",                  0,  14, 0.1,  6.5),
    "rainfall":    ("Rainfall (mm)",            0, 400, 0.1,  200),
}
FERT_FIELDS = {
    "temperature": ("Temperature (°C)",         0,  50, 0.1,  26),
    "humidity":    ("Humidity (%)",             0, 100, 0.1,  52),
    "moisture":    ("Soil moisture (%)",        0, 100, 0.1,  38),
    "nitrogen":    ("Nitrogen",                 0, 100, 1,    37),
    "potassium":   ("Potassium",                0, 100, 1,    0),
    "phosphorous": ("Phosphorous",              0, 100, 1,    0),
}


class InputError(ValueError):
    pass


def num(source, key, spec):
    """Read one numeric field and check it against its allowed range."""
    label, lo, hi = spec[0], spec[1], spec[2]
    raw = source.get(key)
    if raw in (None, ""):
        raise InputError(f"{label} is required.")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InputError(f"{label} must be a number.")
    if not lo <= value <= hi:
        raise InputError(f"{label} must be between {lo} and {hi}.")
    return value


def read_fields(source, fields):
    return {k: num(source, k, spec) for k, spec in fields.items()}


def choice(source, key, options, label):
    value = source.get(key)
    if value not in options:
        raise InputError(f"Please choose a valid {label}.")
    return value


def cleanup_uploads():
    now = time.time()
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isfile(path) and now - os.path.getmtime(path) > UPLOAD_MAX_AGE:
                os.remove(path)
        except OSError:
            pass


def save_upload(file):
    """Validate + store an uploaded image. Returns (path, filename). Raises InputError."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise InputError("Please upload a .jpg, .png, .bmp or .webp image.")
    cleanup_uploads()
    fname = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_DIR, fname)
    file.save(path)
    try:
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, OSError):
        os.remove(path)
        raise InputError("That file is not a readable image.")
    return path, fname


def _load_metrics():
    """Test accuracies shown in the UI, read from results/model_results/*.json."""
    import json
    out = {}
    for key, fname, model in [("crop", "crop_model_results.json", "random_forest"),
                              ("fertilizer", "fertilizer_model_results.json", "decision_tree"),
                              ("disease", "plant_model_results.json", "rf_baseline")]:
        try:
            with open(os.path.join(BASE_DIR, "results", "model_results", fname)) as f:
                out[key] = round(json.load(f)[model]["accuracy"] * 100, 1)
        except (OSError, KeyError, ValueError):
            out[key] = None
    return out


METRICS = _load_metrics()


@app.context_processor
def inject_metrics():
    return {"metrics": METRICS, "model_issues": agent.model_issues}


def ctx(active, **kw):
    return render_template(kw.pop("template"), active=active, **kw)


# ----------------------------------------------------------------- pages
@app.route("/")
def home():
    return render_template("home.html", active="home", backend=agent.disease_backend)


@app.route("/crop", methods=["GET", "POST"])
def crop():
    page = dict(template="crop.html", fields=CROP_FIELDS, form=None, result=None, error=None)
    if request.method == "POST":
        page["form"] = request.form
        try:
            v = read_fields(request.form, CROP_FIELDS)
            result = agent.recommend_crop(v["N"], v["P"], v["K"], v["temperature"],
                                          v["humidity"], v["ph"], v["rainfall"])
            page["error"] = result.get("error")
            page["result"] = None if page["error"] else result
        except InputError as e:
            page["error"] = str(e)
    return ctx("crop", **page)


@app.route("/fertilizer", methods=["GET", "POST"])
def fertilizer():
    page = dict(template="fertilizer.html", fields=FERT_FIELDS, form=None, result=None,
                error=None, soil_types=SOIL_TYPES, crop_types=CROP_TYPES)
    if request.method == "POST":
        page["form"] = request.form
        try:
            v = read_fields(request.form, FERT_FIELDS)
            soil = choice(request.form, "soil_type", SOIL_TYPES, "soil type")
            crop_type = choice(request.form, "crop_type", CROP_TYPES, "crop type")
            result = agent.recommend_fertilizer(v["temperature"], v["humidity"], v["moisture"],
                                                soil, crop_type, v["nitrogen"], v["potassium"],
                                                v["phosphorous"])
            page["error"] = result.get("error")
            page["result"] = None if page["error"] else result
        except InputError as e:
            page["error"] = str(e)
    return ctx("fertilizer", **page)


@app.route("/disease", methods=["GET", "POST"])
def disease():
    page = dict(template="disease.html", result=None, error=None, preview_url=None)
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            page["error"] = "Please choose an image file."
        else:
            try:
                path, fname = save_upload(file)
                page["preview_url"] = url_for("static", filename=f"uploads/{fname}")
                result = agent.detect_disease(path)
                page["error"] = result.get("error")
                page["result"] = None if page["error"] else result
            except InputError as e:
                page["error"] = str(e)
    return ctx("disease", **page)


@app.route("/advisory", methods=["GET", "POST"])
def advisory():
    fields = dict(CROP_FIELDS, moisture=("Soil moisture (%)", 0, 100, 0.1, 35))
    page = dict(template="advisory.html", fields=fields, form=None, result=None, error=None,
                soil_types=SOIL_TYPES, crop_types=CROP_TYPES, preview_url=None)
    if request.method == "POST":
        page["form"] = request.form
        try:
            soil = read_fields(request.form, fields)
            soil["soil_type"] = choice(request.form, "soil_type", SOIL_TYPES, "soil type")
            override = request.form.get("fertilizer_crop_type") or None
            if override:
                choice(request.form, "fertilizer_crop_type", CROP_TYPES, "fertilizer crop type")
            image_path = None
            file = request.files.get("image")
            if file and file.filename:
                image_path, fname = save_upload(file)
                page["preview_url"] = url_for("static", filename=f"uploads/{fname}")
            page["result"] = agent.full_advisory(soil, image_path=image_path,
                                                 fertilizer_crop_type=override)
            page["error"] = page["result"].get("crop_recommendation", {}).get("error")
        except InputError as e:
            page["error"] = str(e)
    return ctx("advisory", **page)


# ------------------------------------------------------------------ JSON API
def _json_error(msg, code=400):
    return jsonify({"error": msg}), code


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "models": {
            "crop": agent.crop_model is not None,
            "fertilizer": agent.fert_model is not None,
            "disease": agent.plant_model is not None or agent.cnn is not None,
        },
        "disease_backend": agent.disease_backend,
        "issues": agent.model_issues,
    })


@app.route("/api/crop", methods=["POST"])
def api_crop():
    try:
        v = read_fields(request.get_json(force=True, silent=True) or {}, CROP_FIELDS)
    except InputError as e:
        return _json_error(str(e))
    return jsonify(agent.recommend_crop(v["N"], v["P"], v["K"], v["temperature"],
                                        v["humidity"], v["ph"], v["rainfall"]))


@app.route("/api/fertilizer", methods=["POST"])
def api_fertilizer():
    body = request.get_json(force=True, silent=True) or {}
    try:
        v = read_fields(body, FERT_FIELDS)
        soil = choice(body, "soil_type", SOIL_TYPES, "soil type")
        crop_type = choice(body, "crop_type", CROP_TYPES, "crop type")
    except InputError as e:
        return _json_error(str(e))
    return jsonify(agent.recommend_fertilizer(v["temperature"], v["humidity"], v["moisture"],
                                              soil, crop_type, v["nitrogen"], v["potassium"],
                                              v["phosphorous"]))


@app.route("/api/disease", methods=["POST"])
def api_disease():
    file = request.files.get("image")
    if not file or not file.filename:
        return _json_error("Send the leaf photo as multipart field 'image'.")
    try:
        path, _ = save_upload(file)
    except InputError as e:
        return _json_error(str(e))
    return jsonify(agent.detect_disease(path))


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    body = request.get_json(force=True, silent=True) or {}
    fields = dict(CROP_FIELDS, moisture=("Soil moisture (%)", 0, 100, 0.1, 35))
    try:
        soil = {k: num(body, k, spec) if k != "moisture" or k in body else 30
                for k, spec in fields.items()}
        soil["soil_type"] = body.get("soil_type", "Loamy")
        choice(soil, "soil_type", SOIL_TYPES, "soil type")
    except InputError as e:
        return _json_error(str(e))
    return jsonify(agent.full_advisory(soil))


@app.errorhandler(413)
def too_large(_e):
    msg = "That image is larger than the 8 MB limit."
    if request.path.startswith("/api/"):
        return _json_error(msg, 413)
    return render_template("disease.html", active="disease", result=None, error=msg,
                           preview_url=None), 413


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return _json_error("Not found", 404)
    return render_template("home.html", active="home", backend=agent.disease_backend), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
