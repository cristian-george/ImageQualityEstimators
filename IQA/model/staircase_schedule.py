from abc import ABC
import tensorflow as tf
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule


class StaircaseSchedule(LearningRateSchedule, ABC):
    def __init__(self, decay_epochs, learning_rates, total_epochs, total_batches):
        super(StaircaseSchedule, self).__init__()
        self.epochs = decay_epochs
        self.lr = tf.constant(learning_rates)
        self.total_epochs = total_epochs
        self.total_batches = total_batches

    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        epoch = step_float / self.total_batches

        lr_0_cond = epoch < self.epochs[0]
        lr_1_cond = tf.logical_and(epoch >= self.epochs[0], epoch < self.epochs[1])
        lr_2_cond = tf.logical_and(epoch >= self.epochs[1], epoch < self.epochs[2])
        lr_3_cond = tf.logical_and(epoch >= self.epochs[2], epoch < self.epochs[3])

        def f(lr_0, lr_1, epoch_0, epoch_1):
            return (lr_1 - lr_0) / (epoch_1 - epoch_0) * (epoch - epoch_0) + lr_0

        # Define the learning rate schedule based on the epoch
        learning_rate = tf.cond(lr_0_cond,
                                lambda: self.lr[0],
                                lambda: tf.cond(lr_1_cond,
                                                lambda: f(self.lr[0], self.lr[1], self.epochs[0], self.epochs[1]),
                                                lambda: tf.cond(lr_2_cond,
                                                                lambda: self.lr[1],
                                                                lambda: tf.cond(lr_3_cond,
                                                                                lambda: f(self.lr[1], self.lr[2],
                                                                                          self.epochs[2],
                                                                                          self.epochs[3]),
                                                                                lambda: self.lr[2]))))

        return learning_rate
