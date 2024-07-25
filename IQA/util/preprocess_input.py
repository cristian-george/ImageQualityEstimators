import tensorflow as tf
from keras.applications.resnet import preprocess_input


def load_and_decode_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    return image


def random_flip_left_right(image):
    image = tf.image.random_flip_left_right(image)
    return image


def random_crop(image, crop_size):
    image_shape = tf.shape(image)[:2]
    offset_height = tf.random.uniform((), minval=0, maxval=image_shape[0] - crop_size[0] + 1, dtype=tf.int32)
    offset_width = tf.random.uniform((), minval=0, maxval=image_shape[1] - crop_size[1] + 1, dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size[0], crop_size[1])
    return image


def crop_5patches(image, target_size):
    image_shape = tf.shape(image)[:2]
    height, width = image_shape[0], image_shape[1]
    center = (height // 2, width // 2)

    patches = [
        tf.image.crop_to_bounding_box(image, 0, 0, target_size[0], target_size[1]),  # Top-left
        tf.image.crop_to_bounding_box(image, 0, width - target_size[1], target_size[0], target_size[1]),  # Top-right
        tf.image.crop_to_bounding_box(image, height - target_size[0], 0, target_size[0], target_size[1]),  # Bottom-left
        tf.image.crop_to_bounding_box(image, height - target_size[0], width - target_size[1], target_size[0],
                                      target_size[1]),  # Bottom-right
        tf.image.crop_to_bounding_box(image, center[0] - target_size[0] // 2, center[1] - target_size[1] // 2,
                                      target_size[0], target_size[1])  # Center
    ]

    return tf.stack(patches)


def load_and_preprocess_input(image_path):
    image = load_and_decode_image(image_path)
    image = preprocess_input(image)

    return image


def crop_and_flip_input(image, target_size=(256, 256), flip_left_right=False):
    if image.shape[:2] != target_size:
        image = random_crop(image, target_size)

    if flip_left_right:
        image = random_flip_left_right(image)

    return image
