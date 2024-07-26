import tensorflow as tf
from keras.applications.resnet import preprocess_input

from util.preprocess_input import load_and_decode_image, random_crop, random_flip_left_right, crop_5patches


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


def flow_train_set_from_dataframe(dataframe, dir_path, batch_size, target_size=None, augment=False):
    image_paths = dataframe['image_name'].apply(lambda image_path: dir_path + "/" + image_path)
    scores = dataframe['MOS']

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, scores))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_input(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.cache()

    if augment and target_size is not None:
        dataset = dataset.map(lambda x, y: (crop_and_flip_input(x, target_size, augment), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(dataset.cardinality(),
                              reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def flow_validation_set_from_dataframe(dataframe, dir_path, batch_size, target_size=None):
    image_paths = dataframe['image_name'].apply(lambda image_path: dir_path + "/" + image_path)
    scores = dataframe['MOS']

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, scores))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_input(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(dataset.cardinality(),
                              reshuffle_each_iteration=True)

    if target_size is not None:
        dataset = dataset.map(lambda x, y: (crop_5patches(x, target_size), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def flow_test_set_from_dataframe(dataframe, dir_path, batch_size, target_size=None):
    image_paths = dataframe['image_name'].apply(lambda image_path: dir_path + "/" + image_path)
    scores = dataframe['MOS']

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, scores))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_input(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    if target_size is not None:
        dataset = dataset.map(lambda x, y: (crop_5patches(x, target_size), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
