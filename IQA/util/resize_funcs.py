import cv2


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
