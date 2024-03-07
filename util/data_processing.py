import pandas as pd
import numpy as np
from PIL import Image
import os

from keras_preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split


# def find_image_score(image_name, csv_file):
#     """
#     Caută scorul unei imagini după numele ei în fișierul CSV.
#
#     :param image_name: Numele imaginii pentru care se caută scorul.
#     :param csv_file: Calea fișierului CSV care conține etichetele.
#     :return: Scorul imaginii dacă este găsit, altfel None.
#     """
#     df = pd.read_csv(csv_file)
#     row_search = df[df['image_name'] == image_name].iloc[0]
#     return row_search['MOS']


def load_and_process_images(directory, csv_file, image_size=(224, 224), output_file="image_label_pairs_comparison_new.txt"):
    """
    Încarcă imaginile și etichetele din setul de date KonIQ-10k.

    :param directory: Calea directorului unde sunt stocate imaginile.
    :param csv_file: Calea fișierului CSV care conține etichetele.
    :param image_size: Dimensiunea la care trebuie redimensionate imaginile.
    :return: Tuple conținând array-uri numpy cu imagini și etichete.
    """
    # Citirea etichetelor din fișierul CSV
    df = pd.read_csv(csv_file)

    # Inițializarea listelor pentru imagini și etichete
    images = []
    labels = []

    with open(output_file, "w") as f:
        for idx, row in df.iterrows():
            # Construirea căii complete a imaginii
            image_path = os.path.join(directory, row['image_name'])

            # Deschiderea și redimensionarea imaginii
            image = Image.open(image_path)
            image = image.resize(image_size)

            # Adăugarea imaginii și a etichetei în listă
            images.append(img_to_array(image))
            labels.append(row['MOS'])

            # Scrierea perechii imagine-etichetă și scorului real în fișier
            # real_score = find_image_score(row['image_name'], csv_file)
            # f.write(f"{row['image_name']} - Extracted Score: {row['MOS']}, Real Score: {real_score}\n")

    # Convertirea listelor în array-uri numpy
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='float32')

    return images, labels


def split_data(images, labels, test_size=0.1):
    return train_test_split(images, labels, test_size=test_size, random_state=42)

