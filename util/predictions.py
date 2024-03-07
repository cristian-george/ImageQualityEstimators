import numpy as np
from PIL import Image
from keras_preprocessing.image import img_to_array


def predict_image_quality(model, image_path):
    mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    new_image = Image.open(image_path)
    new_image = new_image.resize((224, 224))
    new_image = img_to_array(new_image)
    new_image -= mean
    new_image = np.expand_dims(new_image, axis=0)
    predicted_quality = model.predict(new_image)
    return predicted_quality[0][0]
