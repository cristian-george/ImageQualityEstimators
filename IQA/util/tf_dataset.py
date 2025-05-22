from abc import ABC

import tensorflow as tf

from util.tf_preprocess_images import load_and_decode_image


class Dataset(tf.data.Dataset, ABC):
    @staticmethod
    def flow_from_dataframe(dataframe, directory, preprocessing_function, x_col, y_col=None):
        image_paths = dataframe[x_col].apply(lambda image_path: directory + "/" + image_path)

        if y_col and y_col in dataframe.columns:
            scores = dataframe[y_col]
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, scores))
            dataset = dataset.map(lambda x, y: (preprocessing_function(load_and_decode_image(x)), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(image_paths)
            dataset = dataset.map(lambda x: preprocessing_function(load_and_decode_image(x)),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def get_image_shape(dataset):
        image_shapes = set()

        for data in dataset:
            image = data[0] if isinstance(data, tuple) else data
            image_shapes.add(image.shape)

        if len(image_shapes) != 1:
            return None

        target_size = next(iter(image_shapes))
        return target_size
