import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from joblib import dump
import os

# === CONFIGURATION ===
DATA_FOLDER = "train_images"  # Folder where you store training images
PCA_SAVE_PATH = "models/pca_model.joblib"
N_COMPONENTS = 3

# === COLLECT FEATURES ===
X = []
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(DATA_FOLDER, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray, (128, 128))
        hog_feat, _ = hog(roi, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
        X.append(hog_feat)

X = np.array(X)

# === PCA TRAINING ===
pca = PCA(n_components=N_COMPONENTS)
pca.fit(X)
os.makedirs("models", exist_ok=True)
dump(pca, PCA_SAVE_PATH)

print(f"✅ PCA model trained and saved to {PCA_SAVE_PATH}")