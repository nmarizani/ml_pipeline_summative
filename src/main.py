from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil, os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from src.utils.data_upload import save_uploaded_image
from typing import List
from src.db import SessionLocal, UploadedImage
from sqlalchemy.orm import Session
from src.db import SessionLocal, RetrainHistory

app = FastAPI()

@app.get("/")
async def root():
    return {"Vaccine Demand Prediction API is running"}

# Load model at startup
MODEL_PATH = "models/vaccine_demand_model.h5"
CLASS_NAMES = ['low_demand','medium_demand', 'high_demand']  # adjust if you retrain with more

def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

def prepare_image(file_path):
    try:
        # Automatically get model input shape (ignoring batch dimension)
        _, height, width, channels = model.input_shape
        img = image.load_img(file_path, target_size=(height, width))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        img_tensor = prepare_image(file_path)
        preds = model.predict(img_tensor)
        predicted_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))

        return {"prediction": predicted_class, "confidence": confidence}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...), label: str = Form(...)):
    assert label in CLASS_NAMES, f"Label must be one of: {', '.join(CLASS_NAMES)}"
    save_dir = f"data/train/{label}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Save metadata to DB
    db = SessionLocal()
    db.add(UploadedImage(filename=file.filename, label=label))
    db.commit()
    db.close()

    return {"message": f"File uploaded to class '{label}'."}

@app.get("/uploads")
async def get_uploaded_data():
    import pandas as pd
    metadata_path = "uploads/metadata.csv"
    if not os.path.exists(metadata_path):
        return {"message": "No uploaded data yet."}
    
    df = pd.read_csv(metadata_path)
    return df.to_dict(orient="records")

@app.post("/upload-bulk")
async def upload_bulk(files: List[UploadFile] = File(...), labels: List[str] = Form(...)):
    assert len(files) == len(labels), "Each image must have a label."

    temp_file_paths = []
    db = SessionLocal()

    try:
        for file, label in zip(files, labels):
            assert label in CLASS_NAMES, f"Invalid label: {label}"
            temp_path = f"temp_{datetime.now().strftime('%f')}_{file.filename}"
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_file_paths.append((temp_path, label))

        from utils.data_upload import save_bulk_images, should_trigger_retraining
        uploaded_count = save_bulk_images(temp_file_paths, db)

        triggered = False
        if should_trigger_retraining(db):
            await retrain_model()
            triggered = True

        return {
            "message": f"{uploaded_count} images uploaded successfully.",
            "retraining_triggered": triggered
        }

    finally:
        db.close()
        for temp_path, _ in temp_file_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.post("/retrain")
async def retrain_model():
    global model

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        "data/train", target_size=(model.input_shape[1], model.input_shape[2]),
        batch_size=16, class_mode='categorical', subset='training'
    )

    val_gen = datagen.flow_from_directory(
        "data/train", target_size=(model.input_shape[1], model.input_shape[2]),
        batch_size=16, class_mode='categorical', subset='validation'
    )

    new_model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(model.input_shape[1], model.input_shape[2], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(CLASS_NAMES), activation='softmax')
    ])

    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.fit(train_gen, validation_data=val_gen, epochs=5)

    history = new_model.fit(train_gen, validation_data=val_gen, epochs=5)

    version = get_next_model_version()
    model_path = f"models/model_v{version}.h5"
    new_model.save(model_path)

    model = load_model(model_path)

    log_retrain_history(version, history.history)


    return {"Model retrained and updated."}

def get_next_model_version():
    db = SessionLocal()
    latest = db.query(RetrainHistory).order_by(RetrainHistory.id.desc()).first()
    db.close()
    if latest:
        return int(latest.version.replace("v", "")) + 1
    return 1

def log_retrain_history(version: int, history_obj: dict):
    db: Session = SessionLocal()

    try:
        accuracy = history_obj.get("accuracy", [None])[-1]
        notes = f"Accuracy after retrain: {accuracy:.4f}" if accuracy else "Retrained"

        entry = RetrainHistory(
            version=f"v{version}",
            notes=notes
        )
        db.add(entry)
        db.commit()
        print(f"[LOG] Model v{version} retrained and saved to models/model_v{version}.h5")

    except Exception as e:
        print(f"[ERROR] Failed to log retrain history: {e}")
        db.rollback()

    finally:
        db.close()

@app.get("/retrain-history")
async def get_history():
    db = SessionLocal()
    history = db.query(RetrainHistory).order_by(RetrainHistory.id.desc()).all()
    db.close()
    return [
        {"version": h.version, "timestamp": h.timestamp.isoformat(), "notes": h.notes}
        for h in history
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)