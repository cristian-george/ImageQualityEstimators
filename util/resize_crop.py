import cv2
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
