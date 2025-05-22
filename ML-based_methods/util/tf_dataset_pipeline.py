import tensorflow as tf

from model.vars import preprocess_function
from util.tf_dataset import Dataset
from util.tf_preprocess_images import random_crop, random_flip_left_right, crop_5patches


def crop_input(image, target_size):
    image = random_crop(image, target_size)
    return image


def flip_input(image):
    image = random_flip_left_right(image)
    return image


def create_dataset_pipeline(dataframe,
                            directory,
                            subset='train',
                            net_name='resnet50',
                            batch_size=16,
                            target_size=(384, 512, 3),
                            augment=None):
    assert subset in ['train', 'validation', 'test']
    assert net_name in list(preprocess_function.keys())

    has_scores = 'MOS' in dataframe.columns

    if subset in ['train', 'validation'] and not has_scores:
        raise ValueError(f"Missing 'MOS' column for {subset} subset.")

    dataset = Dataset.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        preprocessing_function=preprocess_function[net_name],
        x_col='image_name',
        y_col='MOS' if has_scores else None)

    image_shape = Dataset.get_image_shape(dataset)

    if target_size != image_shape:
        if subset == 'test' and not has_scores:
            dataset = dataset.map(lambda x: crop_5patches(x, target_size),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        elif has_scores:
            if subset == 'train':
                dataset = dataset.map(lambda x, y: (crop_input(x, target_size), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
            else:
                dataset = dataset.map(lambda x, y: (crop_5patches(x, target_size), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)

    if subset == 'train' and augment:
        dataset = dataset.map(lambda x, y: (flip_input(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    if subset == 'train':
        dataset = dataset.shuffle(dataset.cardinality(),
                                  reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)

    if subset == 'validation':
        dataset = dataset.cache()

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, image_shape
