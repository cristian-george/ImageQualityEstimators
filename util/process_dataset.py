import os
import cv2
import pandas as pd
import random


def resize_and_crop(image, target_size=(384, 512)):
    target_h, target_w = target_size
    h, w = image.shape[:2]

    # Resize images to ensure minimum width or height is 512 or 384 respectively
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Perform random cropping to get target size
    h, w = resized_image.shape[:2]
    if abs(w - target_w) < 100 and abs(h - target_h) < 100:
        num_crops = 3
    else:
        num_crops = 5

    crops = []
    for _ in range(num_crops):
        x_start = random.randint(0, abs(w - target_w))
        y_start = random.randint(0, abs(h - target_h))
        cropped_image = resized_image[y_start:y_start + target_h, x_start:x_start + target_w]
        crops.append(cropped_image)

    return crops


def process_images(path_to_images, path_to_labels, output_folder, output_csv):
    # Read the labels CSV file
    labels_df = pd.read_csv(path_to_labels)

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    new_records = []

    for index, row in labels_df.iterrows():
        image_name = row['image_name']
        mos = row['MOS']
        image_path = os.path.join(path_to_images, image_name)

        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            crops = resize_and_crop(image)

            if mos == 5.0:
                if len(crops) == 5:
                    crops.pop()
                    crops.pop()

                crops.pop()

            for i, crop in enumerate(crops):
                new_image_name = f"{os.path.splitext(image_name)[0]}_patch{i + 1}.jpg"
                new_image_path = os.path.join(output_folder, new_image_name)
                cv2.imwrite(new_image_path, crop)
                new_records.append({'image_name': new_image_name, 'MOS': mos})

    # Create a new DataFrame for the new CSV file
    new_df = pd.DataFrame(new_records)

    # Save the new CSV file
    new_df.to_csv(output_csv, index=False)


path_to_images = '../data/LIVE2/images'
path_to_labels = '../data/LIVE2/LIVE2_labels.csv'
output_folder = '../data/LIVE2_Patch/images'
output_csv = '../data/LIVE2_Patch/LIVE2_Patch_labels.csv'

process_images(path_to_images, path_to_labels, output_folder, output_csv)
