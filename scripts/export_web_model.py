"""Export models/pca_model.joblib to docs/model/pca.json for the web version.

The browser cannot read joblib files, so this writes the PCA mean and components as JSON.
Run it again whenever you retrain with train_pca.py:

    python scripts/export_web_model.py
"""
import json
import os

from joblib import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pca = load(os.path.join(ROOT, "models", "pca_model.joblib"))
out = {
    "mean": [round(float(v), 7) for v in pca.mean_],
    "components": [[round(float(v), 7) for v in row] for row in pca.components_],
    "explained": [float(v) for v in pca.explained_variance_ratio_],
}
path = os.path.join(ROOT, "docs", "model", "pca.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(out, f)
print(f"wrote {path}: {len(out['components'])} components of {len(out['mean'])} numbers")
