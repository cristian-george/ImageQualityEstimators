import os
import cv2
import pandas as pd

from resize_crop import resize_and_crop


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
            image = cv2.imread(str(image_path))
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


def main():
    # Generate LIVE2_Patch
    # path_to_images = '../data/LIVE2/images'
    # path_to_labels = '../data/LIVE2/LIVE2_labels.csv'
    # output_folder = '../data/LIVE2_Patch/images'
    # output_csv = '../data/LIVE2_Patch/LIVE2_Patch_labels.csv'

    # Generate LIVEitW_Patch
    path_to_images = '../data/LIVEitW/images'
    path_to_labels = '../data/LIVEitW/LIVEitW_labels.csv'
    output_folder = '../data/LIVEitW_Patch/images'
    output_csv = '../data/LIVEitW_Patch/LIVEitW_Patch_labels.csv'

    process_images(path_to_images, path_to_labels, output_folder, output_csv)


if __name__ == '__main__':
    main()
