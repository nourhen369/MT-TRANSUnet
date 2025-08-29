import tensorflow as tf
import os, datetime, yaml
from datetime import datetime


with open("model/config.yaml") as f:
    config = yaml.safe_load(f)

MASTERLIST_PATH = config["paths"]["masterlist_path"]
SIZE = config["hyperparams"]["size"]
LOGS_DIR = config["logs"]["training_logs"]
os.makedirs(LOGS_DIR, exist_ok=True)

# Timestamped TensorBoard log directory
log_dir = os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=1
    )

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=25, restore_best_weights=True, verbose=1
)