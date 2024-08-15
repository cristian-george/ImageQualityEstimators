from keras.optimizers.schedules.learning_rate_schedule import PiecewiseConstantDecay


def get_step_decay(scheduler_info, steps_per_epoch, num_epochs):
    step_decay = scheduler_info.get('value', {})
    initial_learning_rate = step_decay.get('initial_lr')
    decrease_factor = step_decay.get('factor')
    epoch_per_decay_step = step_decay.get('decay_step')

    num_steps = num_epochs // epoch_per_decay_step

    boundaries = [i * epoch_per_decay_step * steps_per_epoch
                  for i in range(1, num_steps + 1)]

    values = [initial_learning_rate - i * decrease_factor
              for i in range(num_steps + 1)]

    return PiecewiseConstantDecay(boundaries=boundaries,
                                  values=values)
