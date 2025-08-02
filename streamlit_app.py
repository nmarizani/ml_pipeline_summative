import streamlit as st
import requests
import os
from PIL import Image

API_URL = "http://127.0.0.1:8000"
CLASS_NAMES = ["Low_demand", "Medium_demand", "High_demand"]

st.set_page_config(page_title="AfroAI Vaccine Demand Predictor", layout="wide")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Predict Image", "Upload Data", "Retrain", "Retrain History", "Insights"])

# Predict Image
if menu == "Predict Image":
    st.title("Predict Vaccine Demand from Map Image")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded and st.button("Predict"):
        files = {"file": (uploaded.name, uploaded, uploaded.type)}
        response = requests.post(f"{API_URL}/predict", files=files)

        if response.status_code == 200:
            data = response.json()
            st.image(uploaded, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"### Prediction: `{data['prediction']}`")
            st.markdown(f"**Confidence:** {data['confidence']:.2%}")
        else:
            st.error(f"Prediction failed: {response.json().get('error', 'Unknown error')}")

# Upload Data
elif menu == "Upload Data":
    st.title("Upload Training Data")
    mode = st.radio("Choose Upload Type", ["Single Upload", "Bulk Upload"])

    upload_clicked = st.button("Upload")  # One upload button for both modes

    if mode == "Single Upload":
        file = st.file_uploader("Upload single image", type=["png", "jpg", "jpeg"])
        label = st.selectbox("Label", CLASS_NAMES)

        if upload_clicked and file:
            files = {"file": (file.name, file, file.type)}
            data = {"label": label}
            try:
                response = requests.post(f"{API_URL}/upload-data", files=files, data=data)

                try:
                    response_data = response.json()
                except ValueError:
                    st.error(f"Upload failed: Server returned non-JSON response.\n"
                             f"Status: {response.status_code}\n"
                             f"Response: {response.text}")
                else:
                    if response.status_code == 200:
                        st.success(response_data.get("message", "Upload successful!"))
                    else:
                        st.error(f"Upload failed: {response_data.get('error', 'Unknown error')}")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    elif mode == "Bulk Upload":
        files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        labels = st.text_area("Enter comma-separated labels (same order)", "")

        if upload_clicked and files and labels:
            label_list = [l.strip() for l in labels.split(",")]
            if len(label_list) != len(files):
                st.error("Labels count must match number of files.")
            else:
                files_payload = [("files", (f.name, f, f.type)) for f in files]
                data = [("labels", label) for label in label_list]

                try:
                    response = requests.post(f"{API_URL}/upload-bulk", files=files_payload, data=data)

                    try:
                        result = response.json()
                    except ValueError:
                        st.error(f"Bulk upload failed: Server returned non-JSON response.\n"
                                 f"Status: {response.status_code}\n"
                                 f"Response: {response.text}")
                    else:
                        if response.status_code == 200:
                            st.success(result.get("message", "Bulk upload successful!"))
                            if result.get("retraining_triggered"):
                                st.warning("Retraining was automatically triggered.")
                        else:
                            st.error(f"Bulk upload failed: {result.get('error', 'Unknown error')}")
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
# Retrain
elif menu == "Retrain":
    st.title("Retrain Model")

    if st.button("Start Retraining"):
        with st.spinner("Retraining model..."):
            response = requests.post(f"{API_URL}/retrain")
            if response.status_code == 200:
                st.success(response.json().get("message", "Model retrained."))
            else:
                st.error(f"Retrain failed: {response.json().get('error', 'Unknown error')}")

# Retrain History
elif menu == "Retrain History":
    st.title("Retraining Logs")

    response = requests.get(f"{API_URL}/retrain-history")
    if response.status_code == 200:
        history = response.json()
        for entry in history:
            st.markdown(f"""
            - **Version:** `{entry.get("version")}`
            - **Time:** `{entry.get("timestamp")}`
            - **Notes:** `{entry.get("notes")}`
            ---
            """)
    else:
        st.error("Could not fetch retraining history.")

# Insights
elif menu == "Insights":
    st.title("Dataset Insights")
    data_dir = "labeled_data"
    
    if os.path.exists(data_dir):
        stats = {}
        for cls in os.listdir(data_dir):
            class_path = os.path.join(data_dir, cls)
            if os.path.isdir(class_path):
                stats[cls] = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if stats:
            st.bar_chart(stats)
        else:
            st.info("No data found in 'labeled_data' directory.")
    else:
        st.warning(f"Directory `{data_dir}` not found. Make sure it exists or upload data first.")