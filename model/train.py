import os, glob, yaml
from model.training.trainer import Trainer

# --- Load configuration ---
with open("model/config.yaml") as f:
    config = yaml.safe_load(f)

hp = config['hyperparams']
paths = config['paths']

# --- Build train and validation paths ---
train_paths = {
    'x': sorted(glob.glob(os.path.join(paths['train_images'], "*.bmp"))),
    'y1': sorted(glob.glob(os.path.join(paths['train_te_masks'], "*.bmp"))),
    'y2': sorted(glob.glob(os.path.join(paths['train_zp_masks'], "*.bmp"))),
    'y3': sorted(glob.glob(os.path.join(paths['train_icm_masks'], "*.bmp"))),
}

valid_paths = {
    'x': sorted(glob.glob(os.path.join(paths['val_images'], "*.bmp"))),
    'y1': sorted(glob.glob(os.path.join(paths['val_te_masks'], "*.bmp"))),
    'y2': sorted(glob.glob(os.path.join(paths['val_zp_masks'], "*.bmp"))),
    'y3': sorted(glob.glob(os.path.join(paths['val_icm_masks'], "*.bmp"))),
}

# --- Initialize trainer ---
trainer = Trainer(
    train_paths=train_paths,
    valid_paths=valid_paths,
    batch_size=hp['batch_size'],
    n_splits=hp['n_splits'],
    seed=hp['seed'],
    model_name=hp['model_name']
)

# --- Prepare data & run training ---
trainer.prepare_data()
trainer.run_kfold(epochs=hp['epochs'])