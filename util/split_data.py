import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

dataset = "LIVE2_LIVEitW"

dataset_dir = f"../data/{dataset}"
label_file_path = f"{dataset_dir}/{dataset}_labels.csv"
image_dir = f"{dataset_dir}/images"

labels = pd.read_csv(label_file_path)

validation_size = 960
test_size = 1006

train_data, test_data = train_test_split(labels,
                                         test_size=(test_size + validation_size) / len(labels),
                                         random_state=42)

validation_data, test_data = train_test_split(test_data,
                                              test_size=test_size / (test_size + validation_size),
                                              random_state=42)

print("Train set length:", len(train_data))
print("Validation set length:", len(validation_data))
print("Test set length:", len(test_data))

train_data.to_csv(f"{dataset_dir}/train_labels.csv", index=False)
validation_data.to_csv(f"{dataset_dir}/val_labels.csv", index=False)
test_data.to_csv(f"{dataset_dir}/test_labels.csv", index=False)

train_dir = f"{dataset_dir}/train"
val_dir = f"{dataset_dir}/val"
test_dir = f"{dataset_dir}/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

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
