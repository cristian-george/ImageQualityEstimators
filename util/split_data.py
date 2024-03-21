import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

# Calea către fișierul de label-uri și directorul cu imaginile
label_file_path = "../data/KonIQ-10K/koniq10k_scores_and_distributions.csv"
image_dir = "../data/KonIQ-10K/512x384"

# Încarcăm datele folosind pandas
labels = pd.read_csv(label_file_path)

# Împărțim setul de date în antrenare, validare și testare
validation_size = 1000  # Numărul de imagini pentru validare
test_size = 2000  # Numărul de imagini pentru testare

# Împărțim setul de date în antrenare și testare
train_data, test_data = train_test_split(labels, test_size=(test_size + validation_size) / len(labels), random_state=42)

# Împărțim setul de date de testare în validare și testare
validation_data, test_data = train_test_split(test_data, test_size=test_size / (test_size + validation_size),
                                              random_state=42)

# Verificăm dimensiunile seturilor de date
print("Dimensiuni set de antrenare:", len(train_data))
print("Dimensiuni set de validare:", len(validation_data))
print("Dimensiuni set de testare:", len(test_data))

# Salvăm seturile de date în fișiere separate
train_data.to_csv("../data/KonIQ-10K/train_labels.csv", index=False)
validation_data.to_csv("../data/KonIQ-10K/val_labels.csv", index=False)
test_data.to_csv("../data/KonIQ-10K/test_labels.csv", index=False)

# Creăm folderele pentru antrenare, validare și testare
train_dir = "../data/KonIQ-10K/train"
val_dir = "../data/KonIQ-10K/val"
test_dir = "../data/KonIQ-10K/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Mutăm imaginile în folderele corespunzătoare
for index, row in train_data.iterrows():
    image_name = row['image_name']
    source_path = os.path.join(image_dir, image_name)
    destination_path = os.path.join(train_dir, image_name)
    shutil.copyfile(source_path, destination_path)

for index, row in validation_data.iterrows():
    image_name = row['image_name']
    source_path = os.path.join(image_dir, image_name)
    destination_path = os.path.join(val_dir, image_name)
    shutil.copyfile(source_path, destination_path)

for index, row in test_data.iterrows():
    image_name = row['image_name']
    source_path = os.path.join(image_dir, image_name)
    destination_path = os.path.join(test_dir, image_name)
    shutil.copyfile(source_path, destination_path)
