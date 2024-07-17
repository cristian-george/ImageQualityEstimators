import tensorflow as tf
from tensorflow.python.eager.context import LogicalDeviceConfiguration
from tensorflow.python.framework import config


def limit_gpu_memory(memory_limit=1024):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            config.set_logical_device_configuration(gpus[0], [LogicalDeviceConfiguration(memory_limit=memory_limit)])
            print("Using GPU: {}".format(gpus[0]))
        except RuntimeError as e:
            print(e)


def increase_cpu_num_threads(num_threads=1):
    cpus = tf.config.list_physical_devices('CPU')
    if cpus:
        try:
            config.set_inter_op_parallelism_threads(num_threads=num_threads)
            config.set_intra_op_parallelism_threads(num_threads=num_threads)
            print("Using CPU: {} threads".format(num_threads))
        except RuntimeError as e:
            print(e)


def check_gpu_support():
    gpus = tf.config.list_physical_devices('GPU')
    if tf.test.is_built_with_cuda() and tf.test.is_built_with_gpu_support() and gpus:
        print("TensorFlow was built with CUDA support")
        print("TensorFlow was built with cuDNN support")
        return True

    return False
