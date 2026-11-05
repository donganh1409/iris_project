"""Print the Python features of the three training photos, to compare with the web port.

    python scripts/check_web_port.py

Open the web version, choose Sample A, B or C, and the three PCA numbers in
"How it decides" should match these to about two decimals. They differ slightly
because the browser and OpenCV decode JPEGs a little differently.
"""
import os

import cv2
from joblib import load
from skimage.feature import hog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pca = load(os.path.join(ROOT, "models", "pca_model.joblib"))
for name in ("A", "B", "C"):
    img = cv2.imread(os.path.join(ROOT, "train_images", f"{name}.jpg"))
    roi = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (128, 128))
    feats, _ = hog(roi, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    print(name, len(feats), "HOG numbers ->", [round(float(v), 2) for v in pca.transform([feats])[0]])
