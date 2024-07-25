def crop_5patches(image, target_size):
    """
    Crops 5 patches of target size from an image.
    Raises exception if target size is bigger than the image size.
    """
    h, w = image.shape[:2]
    th, tw = target_size

    if tw > w or th > h:
        raise ValueError("Target size must be smaller than the image size")

    # Calculate the center position
    c_x, c_y = w // 2, h // 2

    # Calculate start and end points for the center crop
    center_x_start = max(0, c_x - tw // 2)
    center_x_end = min(w, c_x + tw // 2 + tw % 2)
    center_y_start = max(0, c_y - th // 2)
    center_y_end = min(h, c_y + th // 2 + th % 2)

    left_up = image[0:th, 0:tw]
    right_up = image[0:th, w - tw:w]
    left_down = image[h - th:h, 0:tw]
    right_down = image[h - th:h, w - tw:w]
    center = image[center_y_start:center_y_end, center_x_start:center_x_end]

    return [left_up, right_up, left_down, right_down, center]
