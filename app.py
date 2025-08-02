import streamlit as st
import os, shutil
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from src.db import SessionLocal, UploadedImage, RetrainHistory
from src.utils.data_upload import save_uploaded_image, save_bulk_images, should_trigger_retraining

# Constants
CLASS_NAMES = ['low_demand', 'medium_demand', 'high_demand']
MODEL_DIR = "models"
CURRENT_MODEL_PATH = f"{MODEL_DIR}/vaccine_demand_model.h5"

# Load or fallback to empty model
@st.cache_resource
def load_model_cached():
    return load_model(CURRENT_MODEL_PATH)

model = load_model_cached()

# Sidebar
st.sidebar.title("AfroAI Vaccine Demand Prediction")
menu = st.sidebar.radio("**Navigation**", [
    "Predict Image", "Upload Data", "Insights", "Retrain", "Retrain History"
])

# Helper Functions

def prepare_image_for_model(img):
    _, h, w, _ = model.input_shape
    img = img.resize((w, h))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def get_next_version():
    db = SessionLocal()
    latest = db.query(RetrainHistory).order_by(RetrainHistory.id.desc()).first()
    db.close()
    if latest:
        return int(latest.version.replace("v", "")) + 1
    return 1

def retrain():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    target_size = (model.input_shape[1], model.input_shape[2])

    train_gen = datagen.flow_from_directory(
        "data/train", target_size=target_size, batch_size=16,
        class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        "data/train", target_size=target_size, batch_size=16,
        class_mode='categorical', subset='validation'
    )

    new_model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(CLASS_NAMES), activation='softmax')
    ])

    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = new_model.fit(train_gen, validation_data=val_gen, epochs=5)

    version = get_next_version()
    new_model_path = f"{MODEL_DIR}/model_v{version}.h5"
    new_model.save(new_model_path)

    # Save history
    db = SessionLocal()
    try:
        acc = history.history.get("accuracy", [None])[-1]
        notes = f"Accuracy: {acc:.4f}" if acc else "Retrained"
        db.add(RetrainHistory(version=f"v{version}", notes=notes))
        db.commit()
        st.success(f"Model retrained and saved as version v{version}")
    finally:
        db.close()

# UI Pages

if menu == "Predict Image":
    st.title("Predict Demand from Image")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        try:
            pred_tensor = prepare_image_for_model(img)
            preds = model.predict(pred_tensor)
            pred_class = CLASS_NAMES[np.argmax(preds)]
            confidence = float(np.max(preds))

            st.markdown(f"### Prediction: `{pred_class}`")
            st.markdown(f"**Confidence:** {confidence:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif menu == "Upload Data":
    st.title("Upload Training Data")

    mode = st.radio("Mode", ["Single Upload", "Bulk Upload"])

    if mode == "Single Upload":
        file = st.file_uploader("Upload single image", type=["png", "jpg", "jpeg"])
        label = st.selectbox("Label", CLASS_NAMES)

        if st.button("Upload") and file:
            path = save_uploaded_image(file, label)
            st.success(f"Saved to {path}")
    
    elif mode == "Bulk Upload":
        files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        labels = st.text_area("Enter comma-separated labels (same order)", "")

        if st.button("Upload All") and files and labels:
            label_list = [l.strip() for l in labels.split(",")]
            if len(label_list) != len(files):
                st.error("Labels count must match number of files.")
            else:
                db = SessionLocal()
                try:
                    uploaded = save_bulk_images([(f, l) for f, l in zip(files, label_list)], db)
                    retrain_now = should_trigger_retraining(db)
                    st.success(f"Uploaded {uploaded} images")
                    if retrain_now:
                        st.warning("Retrain threshold met. You may retrain.")
                finally:
                    db.close()

elif menu == "Insights":
    st.title("Dataset Insights")
    class_dirs = [f"data/train/{c}" for c in CLASS_NAMES]
    data = {cls: len(os.listdir(cls)) if os.path.exists(cls) else 0 for cls in CLASS_NAMES}
    df = pd.DataFrame(list(data.items()), columns=["Class", "Images"])
    st.bar_chart(df.set_index("Class"))

elif menu == "Retrain":
    st.title("Retrain Model")

    if st.button("Start Retraining"):
        with st.spinner("Retraining model..."):
            retrain()

elif menu == "Retrain History":
    st.title("Retraining Logs")
    db = SessionLocal()
    history = db.query(RetrainHistory).order_by(RetrainHistory.id.desc()).all()
    db.close()

    for entry in history:
        st.markdown(f"""
        - **Version:** {entry.version}
        - **Time:** {entry.timestamp.strftime("%Y-%m-%d %H:%M")}
        - **Notes:** {entry.notes}
        ---
        """)