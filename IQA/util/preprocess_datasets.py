import tensorflow as tf
from keras.applications.resnet import preprocess_input

from util.preprocess_images import load_and_decode_image, random_crop, random_flip_left_right, crop_5patches


def load_and_preprocess_input(image_path):
    image = load_and_decode_image(image_path)
    image = preprocess_input(image)
    return image


def crop_input(image, target_size=(256, 256)):
    image = random_crop(image, target_size)
    return image


def flip_input(image):
    image = random_flip_left_right(image)
    return image


def get_original_image_shape(dataset):
    image_shapes = set()

    for image, label in dataset:
        image_shape = image.shape[:2]
        image_shapes.add(image_shape)

    if len(image_shapes) != 1:
        return None

    target_size = next(iter(image_shapes))
    return target_size


def flow_from_dataframe(df, dir_path):
    image_paths = df['image_name'].apply(lambda image_path: dir_path + "/" + image_path)
    scores = df['MOS']

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, scores))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_input(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def create_train_set_pipeline(df, dir_path, batch_size, target_size, augment=False):
    dataset = flow_from_dataframe(df, dir_path)
    dataset = dataset.cache()

    image_shape = get_original_image_shape(dataset)

    if target_size != image_shape:
        dataset = dataset.map(lambda x, y: (crop_input(x, target_size), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(lambda x, y: (flip_input(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(dataset.cardinality())

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, image_shape


def create_validation_set_pipeline(df, dir_path, batch_size, target_size):
    dataset = flow_from_dataframe(df, dir_path)

    image_shape = get_original_image_shape(dataset)

    if target_size != image_shape:
        dataset = dataset.map(lambda x, y: (crop_5patches(x, target_size), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(dataset.cardinality())

    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, image_shape


def create_test_set_pipeline(df, dir_path, batch_size, target_size):
    dataset = flow_from_dataframe(df, dir_path)

    image_shape = get_original_image_shape(dataset)

    if target_size != image_shape:
        dataset = dataset.map(lambda x, y: (crop_5patches(x, target_size), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, image_shape
