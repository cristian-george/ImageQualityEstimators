from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay


def get_exponential_decay(scheduler_info, steps_per_epoch, num_epochs):
    exponential_decay = scheduler_info.get('value', {})
    initial_learning_rate = exponential_decay.get('initial_lr')
    final_learning_rate = exponential_decay.get('final_lr')
    staircase = exponential_decay.get('staircase')

    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)

    return ExponentialDecay(initial_learning_rate=initial_learning_rate,
                            decay_steps=steps_per_epoch,
                            decay_rate=decay_rate,
                            staircase=staircase)
