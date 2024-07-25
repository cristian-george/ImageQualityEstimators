import os

import cv2
from tqdm import tqdm

from util.resize_funcs import white_fill


def white_fill_images(input_folder, output_folder, target_size):
    """
    Pads all images in the input_folder and saves them to the output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, filename in tqdm(enumerate(os.listdir(input_folder)), desc="White fill images",
                            total=len(os.listdir(input_folder))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            padded_image = white_fill(image, target_size)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, padded_image)
