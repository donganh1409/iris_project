import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load, dump
import os

# --- Paths ---
DB_PATH = "data/iris_features.joblib"
IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Load database ---
if os.path.exists(DB_PATH):
    database = load(DB_PATH)
else:
    database = {}

# --- Load PCA ---
try:
    pca = load("models/pca_model.joblib")
except:
    st.error("❌ PCA model not found. Place it in 'models/pca_model.joblib'")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🔍 Verify Iris", "➕ Add New Identity", "🗂 Show All Identities"])

# --- Sidebar: Registered Identities ---
st.sidebar.markdown("### 👁 Registered Identities")
if database:
    for name in sorted(database.keys()):
        col1, col2 = st.sidebar.columns([4, 1])
        col1.markdown(f"- {name}")
        if col2.button("❌", key=f"del_{name}_sidebar"):
            del database[name]
            img_path = os.path.join(IMAGE_DIR, f"{name}.jpg")
            if os.path.exists(img_path):
                os.remove(img_path)
            dump(database, DB_PATH)
            st.experimental_rerun()
else:
    st.sidebar.markdown("ℹ️ No identities yet.")

# --- Preprocessing ---
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (128, 128))
    hog_features, _ = hog(roi, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return pca.transform([hog_features])[0]

# --- Match Function ---
def match(features, threshold=0.7):
    if not database:
        return "Unknown", 0.0
    scores = {label: cosine_similarity([features], [vec])[0][0] for label, vec in database.items()}
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] > threshold else ("Unknown", scores[best])

# --- Page: Verify Iris ---
if page == "🔍 Verify Iris":
    st.title("🔍 Iris Verification")

    method = st.radio("Image Source", ["Webcam", "Upload"])
    image = None

    if method == "Webcam":
        picture = st.camera_input("Take photo")
        if picture:
            img_array = np.asarray(bytearray(picture.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        if file:
            img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is not None:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input Image")
        if st.button("Scan Iris"):
            try:
                feats = preprocess(image)
                name, score = match(feats)
                if name == "Unknown":
                    st.warning(f"❌ No match (Score: {score:.2f})")
                else:
                    st.success(f"✅ Matched: {name} (Score: {score:.2f})")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# --- Page: Add New Identity ---
elif page == "➕ Add New Identity":
    st.title("➕ Add New Iris Identity")
    name = st.text_input("Name/ID").strip().lower()
    img = st.file_uploader("Upload close-up iris image", type=["png", "jpg", "jpeg"])

    if name and img:
        img_array = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Preview")

        if name in database:
            st.warning(f"⚠️ '{name}' already exists.")
        elif st.button("➕ Save"):
            try:
                feats = preprocess(frame)
                database[name] = feats
                cv2.imwrite(os.path.join(IMAGE_DIR, f"{name}.jpg"), frame)
                dump(database, DB_PATH)
                st.success(f"✅ '{name}' saved successfully.")
            except Exception as e:
                st.error(f"❌ Failed: {e}")

# --- Page: Show All Identities ---
elif page == "🗂 Show All Identities":
    st.title("🗂 Registered Iris Identities")
    if not database:
        st.info("No identities found.")
    else:
        for name in sorted(database.keys()):
            st.subheader(name)
            img_path = os.path.join(IMAGE_DIR, f"{name}.jpg")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{name}'s iris image", width=250)
            else:
                st.warning("No image found.")
            if st.button(f"❌ Delete '{name}'", key=f"del_{name}_page"):
                del database[name]
                if os.path.exists(img_path):
                    os.remove(img_path)
                dump(database, DB_PATH)
                st.experimental_rerun()