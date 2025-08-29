import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import KFold
from glob import glob

from model.data.dataset import tf_dataset_multi_with_cls, filter_dataset_by_outcome, SIZE
from model.data.sampler import undersample_majority_class
from model.network.build import build_transunet
from model.training.metrics import dice_coef_multiclass, iou_multiclass, dice_loss
from model.training.callbacks import *
from model.configs.paths import MODELS_DIR


class Trainer:
    def __init__(self, train_paths, valid_paths, batch_size=8, n_splits=5, seed=42, model_name="v1"):
        """
        Trainer for multi-task TRANSU-Net (segmentation + classification).

        Args:
            train_paths (dict): Paths to training images and masks.
            valid_paths (dict): Paths to validation images and masks.
            batch_size (int): Batch size for training.
            n_splits (int): Number of folds for K-Fold CV.
            seed (int): Random seed for reproducibility.
        """
        self.train_paths = train_paths
        self.valid_paths = valid_paths
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.seed = seed
        self.model_name = model_name

    def prepare_data(self):
        """Load, filter, and undersample training and validation data."""
        # --- Training ---
        filtered_train = filter_dataset_by_outcome(
            self.train_paths['x'],
            self.train_paths['y1'],
            self.train_paths['y2'],
            self.train_paths['y3']
        )

        print("Training data before undersampling:", Counter(filtered_train[4]))

        # Undersample majority class
        undersampled_train = undersample_majority_class(*filtered_train)
        print("Training data after undersampling:", Counter(undersampled_train[4]))

        self.train_data = undersampled_train

        # --- Validation ---
        filtered_valid = filter_dataset_by_outcome(
            self.valid_paths['x'],
            self.valid_paths['y1'],
            self.valid_paths['y2'],
            self.valid_paths['y3']
        )
        self.valid_data = filtered_valid
        print("Validation data:", Counter(filtered_valid[4]))

    def create_tf_dataset(self, X, y1, y2, y3, cls_labels, shuffle=True):
        return tf_dataset_multi_with_cls(X, y1, y2, y3, cls_labels, batch_size=self.batch_size, shuffle=shuffle)

    def run_kfold(self, epochs=200):
        """K-Fold cross-validation training with per-fold metrics logging."""
        X = np.array(self.train_data[0])
        y1 = np.array(self.train_data[1])
        y2 = np.array(self.train_data[2])
        y3 = np.array(self.train_data[3])
        y_cls = np.array(self.train_data[4]).astype(np.float32)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        fold_no = 1
        val_scores = []
        all_fold_histories = []

        for train_idx, val_idx in kf.split(X):
            print(f"\n--- Fold {fold_no} ---")

            train_ds_fold = self.create_tf_dataset(
                X[train_idx], y1[train_idx], y2[train_idx], y3[train_idx], y_cls[train_idx], shuffle=True
            )
            val_ds_fold = self.create_tf_dataset(
                X[val_idx], y1[val_idx], y2[val_idx], y3[val_idx], y_cls[val_idx], shuffle=False
            )

            model = build_transunet(return_attention=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss={
                    "segmentation": dice_loss,
                    "classification": tf.keras.losses.BinaryCrossentropy(from_logits=False)
                },
                loss_weights={"segmentation": 1.0, "classification": 2.5},
                metrics={
                    "segmentation": [dice_coef_multiclass, iou_multiclass],
                    "classification": [
                        tf.keras.metrics.AUC(name="auprc", curve="PR"),
                        tf.keras.metrics.Precision(name="precision"),
                        tf.keras.metrics.Recall(name="recall"),
                        tf.keras.metrics.BinaryAccuracy(name="accuracy")
                    ]
                }
            )

            callbacks = [reduce_lr, early_stop]

            history = model.fit(
                train_ds_fold,
                validation_data=val_ds_fold,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )

            # Append fold metrics to history list
            fold_history = pd.DataFrame(history.history)
            fold_history["fold"] = fold_no
            all_fold_histories.append(fold_history)

            # Evaluate and store validation scores
            scores = model.evaluate(val_ds_fold, verbose=0)
            print(f"Fold {fold_no} scores: {scores}")
            val_scores.append(scores)

            # Save fold-specific model
            MODEL_NAME = os.path.join(MODELS_DIR, f"{self.model_name}_fold{fold_no}")
            model.save(MODEL_NAME)

            fold_no += 1

        # Combine all fold histories and save as CSV
        all_histories_df = pd.concat(all_fold_histories, ignore_index=True)
        csv_path = os.path.join(MODELS_DIR, f"{self.model_name}_kfold_metrics.csv")
        all_histories_df.to_csv(csv_path, index=False)
        print(f"\nSaved per-fold metrics to {csv_path}")

        # Display aggregate results
        val_scores = np.array(val_scores)
        print("\n=== K-Fold Cross-Validation Results ===")
        print("Mean:", val_scores.mean(axis=0))
        print("Std:", val_scores.std(axis=0))


if __name__ == "__main__":
    import os
    from glob import glob
    from model.configs.paths import (
        TRAIN_IMAGES, TRAIN_TE_MASKS, TRAIN_ZP_MASKS, TRAIN_ICM_MASKS,
        VAL_IMAGES, VAL_TE_MASKS, VAL_ZP_MASKS, VAL_ICM_MASKS
    )

    train_paths = {
        'x': sorted(glob(os.path.join(TRAIN_IMAGES, "*.bmp"))),
        'y1': sorted(glob(os.path.join(TRAIN_TE_MASKS, "*.bmp"))),
        'y2': sorted(glob(os.path.join(TRAIN_ZP_MASKS, "*.bmp"))),
        'y3': sorted(glob(os.path.join(TRAIN_ICM_MASKS, "*.bmp"))),
    }

    valid_paths = {
        'x': sorted(glob(os.path.join(VAL_IMAGES, "*.bmp"))),
        'y1': sorted(glob(os.path.join(VAL_TE_MASKS, "*.bmp"))),
        'y2': sorted(glob(os.path.join(VAL_ZP_MASKS, "*.bmp"))),
        'y3': sorted(glob(os.path.join(VAL_ICM_MASKS, "*.bmp"))),
    }

    # Initialize and run trainer
    trainer = Trainer(train_paths, valid_paths, batch_size=8, n_splits=5)
    trainer.prepare_data()
    trainer.run_kfold(epochs=200)