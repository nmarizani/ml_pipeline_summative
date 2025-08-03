import streamlit as st
import requests
import os
from PIL import Image
import pandas as pd

API_URL = "https://vaccine-demand-ml-api.onrender.com"
CLASS_NAMES = ["low_demand", "medium_demand", "high_demand"]

st.set_page_config(page_title="AfroAI Vaccine Demand Predictor", layout="wide")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Status", "Predict Image", "Upload Data", "Retrain", "Retrain History", "Insights"])

# Server Status
if menu == "Status":
    st.title("Model & Server Status")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            st.success("API is running!")
            st.json(response.json())
        else:
            st.error("API did not respond successfully.")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API: {e}")

# Predict Image
elif menu == "Predict Image":
    st.title("Predict Vaccine Demand from Map Image")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded and st.button("Predict"):
        files = {"file": (uploaded.name, uploaded, uploaded.type)}
        try:
            response = requests.post(f"{API_URL}/predict", files=files)
            data = response.json()
            if response.status_code == 200:
                st.image(uploaded, caption="Uploaded Image", use_column_width=True)
                st.markdown(f"### Prediction: `{data['prediction']}`")
                st.markdown(f"**Confidence:** {data['confidence']:.2%}")
            else:
                st.error(f"Prediction failed: {data.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# Upload Data
elif menu == "Upload Data":
    st.title("Upload Training Data")
    mode = st.radio("Choose Upload Type", ["Single Upload", "Bulk Upload"])
    upload_clicked = st.button("Upload")

    if mode == "Single Upload":
        file = st.file_uploader("Upload single image", type=["png", "jpg", "jpeg"])
        label = st.selectbox("Label", CLASS_NAMES)

        if upload_clicked and file:
            try:
                response = requests.post(
                    f"{API_URL}/upload-data",
                    files={"file": (file.name, file, file.type)},
                    data={"label": label},
                )
                if response.status_code == 200:
                    st.success(response.json().get("message", "Upload successful!"))
                else:
                    st.error(f"Upload failed: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Upload request failed: {e}")

    elif mode == "Bulk Upload":
        files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        labels = st.text_area("Enter comma-separated labels (same order)")

        if upload_clicked and files and labels:
            label_list = [l.strip() for l in labels.split(",")]
            if len(label_list) != len(files):
                st.error("Labels count must match number of files.")
            else:
                files_payload = [("files", (f.name, f, f.type)) for f in files]
                data = [("labels", l) for l in label_list]

                try:
                    response = requests.post(f"{API_URL}/upload-bulk", files=files_payload, data=data)
                    result = response.json()
                    if response.status_code == 200:
                        st.success(result.get("message", "Bulk upload successful!"))
                        if result.get("retraining_triggered"):
                            st.warning("Retraining was automatically triggered.")
                    else:
                        st.error(f"Upload failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Bulk upload failed: {e}")

# Retrain
elif menu == "Retrain":
    st.title("Manual Model Retraining")

    if st.button("Start Retraining"):
        with st.spinner("Training model..."):
            try:
                response = requests.post(f"{API_URL}/retrain")
                if response.status_code == 200:
                    st.success(response.json().get("message", "Model retrained."))
                else:
                    st.error(response.json().get("error", "Retrain failed"))
            except Exception as e:
                st.error(f"Retraining failed: {e}")

# Retrain History
elif menu == "Retrain History":
    st.title("Retraining Logs")

    try:
        response = requests.get(f"{API_URL}/retrain-history")
        history = response.json()
        if response.status_code == 200:
            for entry in history:
                st.markdown(f"""
                - **Version:** `{entry.get("version")}`
                - **Notes:** {entry.get("notes", '')}
                ---
                """)
        else:
            st.error("Could not load history.")
    except Exception as e:
        st.error(f"Error: {e}")

# Dataset Insights
elif menu == "Insights":
    st.title("Dataset Insights")
    data_dir = "data/train"

    if os.path.exists(data_dir):
        stats = {}
        for cls in os.listdir(data_dir):
            cls_path = os.path.join(data_dir, cls)
            if os.path.isdir(cls_path):
                stats[cls] = len([f for f in os.listdir(cls_path) if f.endswith((".png", ".jpg", ".jpeg"))])

        if stats:
            st.subheader("Class Distribution")
            df = pd.DataFrame.from_dict(stats, orient='index', columns=['Count'])
            st.bar_chart(df)

            st.subheader("Sample Images")
            for cls in stats:
                cls_path = os.path.join(data_dir, cls)
                imgs = [f for f in os.listdir(cls_path) if f.endswith((".png", ".jpg", ".jpeg"))][:3]
                if imgs:
                    st.markdown(f"**{cls.capitalize()}**")
                    cols = st.columns(len(imgs))
                    for i, img_file in enumerate(imgs):
                        img_path = os.path.join(cls_path, img_file)
                        with cols[i]:
                            st.image(Image.open(img_path), use_column_width=True)
        else:
            st.warning("No labeled data found yet.")
    else:
        st.error("Directory `data/train` does not exist. Upload data first.")