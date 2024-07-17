import cv2
import random


def resize_image(image, target_size=(384, 512), interpolation=cv2.INTER_CUBIC):
    target_h, target_w = target_size
    height, width = image.shape[:2]

    # Resize images to ensure minimum width or height is 512 or 384 respectively
    scale = max(target_w / width, target_h / height)
    new_w, new_h = int(width * scale), int(height * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    return resized_image


def resize_crop_5patches(image, target_size=(384, 512), interpolation=cv2.INTER_CUBIC):
    height, width = image.shape[:2]
    target_width, target_height = target_size

    if target_width > width or target_height > height:
        raise ValueError("Target size must be smaller than the image dimensions")

    center_x, center_y = width // 2, height // 2

    left_up = image[0:target_height, 0:target_width]
    right_up = image[0:target_height, width - target_width:width]
    left_down = image[height - target_height:height, 0:target_width]
    right_down = image[height - target_height:height, width - target_width:width]
    center = image[
             center_y - target_height // 2:center_y + target_height // 2,
             center_x - target_width // 2:center_x + target_width // 2
             ]

    return [left_up, right_up, left_down, right_down, center]


def resize_and_crop(image, target_size=(384, 512), interpolation=cv2.INTER_CUBIC):
    resized_image = resize_image(image, target_size, interpolation)

    # Perform random cropping to get target size
    target_h, target_w = target_size
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
