from keras.callbacks import TensorBoard


def get_tensorboard_callback(callbacks_info):
    tensorboard_info = callbacks_info.get('tensorboard', {})
    log_dir = tensorboard_info.get('log_dir', '')
    histogram_freq = tensorboard_info.get('histogram_freq')

    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                       histogram_freq=histogram_freq)

    return tensorboard_callback
