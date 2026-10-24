"""
retrain_models.py
-------------------
Rebuilds the CROP and FERTILIZER models with the scikit-learn version installed
on THIS computer. Use it when the web app shows a "Model loading problem" banner
(models saved with scikit-learn 1.8.0, but a different version is installed).

    python retrain_models.py

The leaf-disease model needs the original leaf photos (not included in the
project), so it cannot be rebuilt here. If it is also flagged, install
scikit-learn 1.8.0 instead:  pip install scikit-learn==1.8.0   (Python 3.11+)
"""
import os
import sys
import sklearn

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "preprocessing", "models"):
    sys.path.insert(0, os.path.join(ROOT, sub))
os.chdir(ROOT)

print(f"Rebuilding models with scikit-learn {sklearn.__version__} ...\n")

import crop_preprocessing, fertilizer_preprocessing      # noqa: E402
import crop_model, fertilizer_model                       # noqa: E402

crop_preprocessing.preprocess()
crop_model.run()
fertilizer_preprocessing.preprocess()
fertilizer_model.run()

# keep the saved models small so they fit on GitHub (files must stay under 25 MB for browser upload)
import glob, joblib                                       # noqa: E402
for f in glob.glob(os.path.join(ROOT, "models", "saved", "*.joblib")):
    joblib.dump(joblib.load(f), f, compress=3)

print("\nDone. Crop + fertilizer models were rebuilt and saved to models/saved/.")
print("Restart the web app:  python app/app.py")
