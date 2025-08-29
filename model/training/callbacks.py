import tensorflow as tf


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=1
    )

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=25, restore_best_weights=True, verbose=1
)