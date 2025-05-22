import os

import cv2
from tqdm import tqdm


def white_fill(image, target_size):
    """
    Adds white padding to the image until it reaches the target size.
    """
    h, w = image.shape[:2]
    th, tw = target_size

    # Calculate the padding amounts
    pad_top = max((th - h) // 2, 0)
    pad_bottom = max(th - h - pad_top, 0)
    pad_left = max((tw - w) // 2, 0)
    pad_right = max(tw - w - pad_left, 0)

    # Add padding to the image
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                      value=[255, 255, 255])

    return padded_image


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
