import os

import cv2
from tqdm import tqdm


def resize(image, target_size, interpolation=cv2.INTER_NEAREST):
    """
    Resizes image to ensure minimum height or width is th or tw respectively.
    Note: th, tw = target_size
    """
    h, w = image.shape[:2]
    th, tw = target_size

    scale = max(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    return resized_image


def resize_images(input_folder, output_folder, target_size):
    """
    Resizes all images in the input_folder and saves them to the output_folder.
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

            padded_image = resize(image, target_size, interpolation=cv2.INTER_CUBIC)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, padded_image)
