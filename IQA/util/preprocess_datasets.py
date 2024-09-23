from abc import ABC

import tensorflow as tf

from model.vars import preprocess_function
from util.preprocess_images import load_and_decode_image, random_crop, random_flip_left_right, crop_5patches


def load_image(image_path):
    image = load_and_decode_image(image_path)
    return image


def crop_input(image, target_size):
    image = random_crop(image, target_size)
    return image


def flip_input(image):
    image = random_flip_left_right(image)
    return image


class Dataset(tf.data.Dataset, ABC):
    @staticmethod
    def flow_from_dataframe(dataframe, directory, preprocessing_function, x_col, y_col):
        image_paths = dataframe[x_col].apply(lambda image_path: directory + "/" + image_path)
        scores = dataframe[y_col]

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, scores))
        dataset = dataset.map(lambda x, y: (preprocessing_function(load_image(x)), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def get_image_shape(dataset):
        image_shapes = set()

        for image, label in dataset:
            image_shapes.add(image.shape)

        if len(image_shapes) != 1:
            return None

        target_size = next(iter(image_shapes))
        return target_size


def create_dataset_pipeline(dataframe,
                            directory,
                            subset='train',
                            net_name='resnet50',
                            batch_size=16,
                            target_size=(384, 512, 3),
                            augment=None):
    """
        Creates a dataset pipeline for training, validation, or testing purposes using TensorFlow's Dataset API
        and a DataFrame of images. This function processes images and applies preprocessing, augmentation, and
        caching techniques based on the provided parameters.

        Parameters:
        -----------
        dataframe : pandas.DataFrame
            A DataFrame containing the dataset information. It must include a column 'image_name' for image paths
            and a column 'MOS' for the target labels (Mean Opinion Score).

        directory : str
            Path to the directory where the images are stored.

        subset : str, optional (default='train')
            Specifies the subset of the dataset pipeline to create. Can be one of the following:
            - 'train': Used for training. Will include data augmentation, shuffling, and caching.
            - 'validation': Used for validation. Will include caching but not augmentation or shuffling.
            - 'test': Used for testing. No augmentation or shuffling.

        net_name : str, optional (default='resnet50')
            The name of the neural network model being used. This determines the preprocessing function applied to
            the images. The function retrieves the preprocessing function using `preprocess_function[net_name]`.

        batch_size : int, optional (default=16)
            The number of images to include in each batch for the dataset.

        target_size : tuple, optional (default=(384, 512, 3))
            The target size to resize the images to. The tuple should be in the format (height, width, channels).

        augment : function, optional (default=None)
            If not None, specifies an augmentation function to apply during training. This function is applied
            only when `subset='train'`. Flipping images left-right is applied.

        Returns:
        --------
        dataset : tf.data.Dataset
            A TensorFlow Dataset object representing the pipeline, ready for training, validation, or testing.

        image_shape : tuple
            The shape of the images in the dataset after any resizing or cropping. This may be different from the
            original image shape.

        Notes:
        ------
        - If the `subset` is 'train', the dataset will be cached for better performance, shuffled, and augmented
          (if `augment` is provided).
        - If the `subset` is 'validation', the dataset will be cached for performance improvement but no
          augmentation or shuffling is applied.
        - If the `subset` is 'test', no augmentation or caching is applied.
        - The `target_size` parameter allows resizing or cropping to match the input size expected by the model.
          If the actual image shape differs from the `target_size`, either cropping or 5-patch cropping is
          applied, depending on whether the subset is for training or validation/testing.
        - Preprocessing is handled by the network-specific preprocessing function retrieved via
          `preprocess_function[net_name]`.
        """
    assert subset in ['train', 'validation', 'test']
    assert net_name in list(preprocess_function.keys())
    assert batch_size >= 2

    dataset = Dataset.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        preprocessing_function=preprocess_function[net_name],
        x_col='image_name',
        y_col='MOS')

    if subset == 'train':
        dataset = dataset.cache()

    image_shape = Dataset.get_image_shape(dataset)

    if target_size != image_shape:
        if subset == 'train':
            dataset = dataset.map(lambda x, y: (crop_input(x, target_size), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(lambda x, y: (crop_5patches(x, target_size), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)

    if subset == 'train':
        if augment:
            dataset = dataset.map(lambda x, y: (flip_input(x), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.shuffle(dataset.cardinality(),
                                  reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)

    if subset == 'validation':
        dataset = dataset.cache()

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, image_shape
