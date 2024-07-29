import os

from keras.callbacks import ModelCheckpoint, EarlyStopping


def get_model_checkpoint_callbacks(callbacks_info):
    model_checkpoint_info = callbacks_info.get('model_checkpoint', {})
    ckpt_dir = model_checkpoint_info.get('ckpt_dir', '')

    ckpts_info = model_checkpoint_info.get('ckpts', [])
    for ckpt_info in ckpts_info:
        monitor = ckpt_info.get('monitor', '')
        mode = ckpt_info.get('mode', '')
        save_best_only = ckpt_info.get('save_best_only')
        save_weights_only = ckpt_info.get('save_weights_only')

        path = os.path.join(ckpt_dir, monitor)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        model_checkpoint_callback = ModelCheckpoint(os.path.join(str(path), 'best_model_{epoch:02d}.h5'),
                                                    monitor=monitor,
                                                    mode=mode,
                                                    save_best_only=save_best_only,
                                                    save_weights_only=save_weights_only)

        early_stopping_callback = EarlyStopping(monitor=monitor,
                                                mode=mode,
                                                patience=20)

        yield model_checkpoint_callback
        yield early_stopping_callback
