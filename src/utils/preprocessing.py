from PIL import Image
import numpy as np

def preprocess_image(image, target_size=(64, 64)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape((1, *target_size, 3))
    return image_array