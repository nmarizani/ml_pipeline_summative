import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils.model import load_trained_model
import os

# Define image size used during training
IMAGE_SIZE = (64, 64)
CLASSES = ['high_demand', 'medium_demand', 'low_demand']

def prepare_image(image_path):
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image_path):
    model = load_trained_model()
    image = prepare_image(image_path)
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    predicted_label = CLASSES[predicted_index]
    confidence = float(np.max(prediction))
    return predicted_label, confidence
