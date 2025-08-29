import os, re, cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from model.configs.train import SIZE
from model.configs.paths import MASTERLIST_PATH


def read_image_tf(path: bytes) -> np.ndarray:
    """
    Reads and preprocesses an image from disk, then returns a normalized image array of shape (SIZE, SIZE, 3).
    """
    try:
        path = path.decode()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        x = cv2.imread(path, cv2.IMREAD_COLOR)
        if x is None:
            raise ValueError(f"Failed to read image: {path}")

        x = cv2.resize(x, (SIZE, SIZE))
        x = x / 255.0
        return x.astype(np.float32)
    except Exception as e:
        print(f"[ERROR] read_image_tf failed for {path}: {e}")
        return np.zeros((SIZE, SIZE, 3), dtype=np.float32)


def read_mask_tf(path: bytes) -> np.ndarray:
    """
    Reads and preprocesses a binary segmentation mask, then returns a binary mask of shape (SIZE, SIZE, 1).
    """
    try:
        path = path.decode()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask file not found: {path}")

        y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if y is None:
            raise ValueError(f"Failed to read mask: {path}")

        y = cv2.resize(y, (SIZE, SIZE))
        y[y != 0] = 255
        y = y / 255.0
        y = (y > 0.5).astype(np.float32)
        y = np.expand_dims(y, axis=-1)
        return y
    except Exception as e:
        print(f"[ERROR] read_mask_tf failed for {path}: {e}")
        return np.zeros((SIZE, SIZE, 1), dtype=np.float32)


def read_multiclass_mask_tf(te_path: bytes, zp_path: bytes, icm_path: bytes) -> np.ndarray:
    """
    Reads and concatenates multiple binary masks into a multi-class mask of shape (SIZE, SIZE, 3).
    """
    try:
        te = read_mask_tf(te_path)
        zp = read_mask_tf(zp_path)
        icm = read_mask_tf(icm_path)
        mask = np.concatenate([te, zp, icm], axis=-1)
        return mask
    except Exception as e:
        print(f"[ERROR] read_multiclass_mask_tf failed: {e}")
        return np.zeros((SIZE, SIZE, 3), dtype=np.float32)


def tf_parse_multi_with_cls(x_path, te_path, zp_path, icm_path, cls_label):
    """
    TF wrapper to parse images, segmentation masks, and classification labels.
    
    Returns:
        tuple: (image, (multi-class mask, classification label))
    """
    def _parse(x_path, te_path, zp_path, icm_path, cls_label):
        x = read_image_tf(x_path)
        y = read_multiclass_mask_tf(te_path, zp_path, icm_path)
        return x, y, cls_label

    x, y, cls_label = tf.numpy_function(
        _parse,
        [x_path, te_path, zp_path, icm_path, cls_label],
        [tf.float32, tf.float32, tf.float32]
    )

    x.set_shape([SIZE, SIZE, 3])
    y.set_shape([SIZE, SIZE, 3]) # 3 channels: TE, ZP, ICM
    cls_label.set_shape([])

    return x, (y, cls_label)


def tf_dataset_multi_with_cls(x_paths, te_paths, zp_paths, icm_paths, cls_labels, batch_size=16, shuffle=True):
    """
    Builds a TensorFlow dataset pipeline for multi-task learning (segmentation + classification).
    """
    try:
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_paths, te_paths, zp_paths, icm_paths, cls_labels)
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(tf_parse_multi_with_cls, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        print(f"[INFO] Dataset built with {len(x_paths)} samples, batch size {batch_size}")
        return dataset
    except Exception as e:
        print(f"[ERROR] Failed to build dataset: {e}")
        return tf.data.Dataset.from_tensor_slices(())


def get_image_to_outcome() -> dict:
    """
    Reads the master list Excel file and builds a mapping from file name to outcome.
    
    Returns:
        dict: Mapping {file_name: outcome}
    """
    try:
        if not os.path.exists(MASTERLIST_PATH):
            raise FileNotFoundError(f"Masterlist file not found: {MASTERLIST_PATH}")

        df = pd.read_excel(MASTERLIST_PATH)
        if "Outcome" not in df.columns or "File Name" not in df.columns:
            raise ValueError("Excel file missing required columns: 'Outcome' or 'File Name'")

        df = df[df['Outcome'] != 2]  # filter out outcome == 2
        mapping = dict(zip(df['File Name'], df['Outcome']))
        print(f"[INFO] Loaded {len(mapping)} image→outcome mappings")
        return mapping
    except Exception as e:
        print(f"[ERROR] Failed to read masterlist: {e}")
        return {}


def get_outcome_from_path(path: str, image_to_outcome: dict):
    """
    Retrieves the outcome label from an image path using masterlist mapping.
    
    Returns:
        float|None: Outcome label or None if missing.
    """
    try:
        fname = os.path.basename(path)
        base_fname = os.path.splitext(fname)[0]
        base_key = re.sub(r'_\d+$', '', base_fname)  # strip augmentation suffix
        outcome = image_to_outcome.get(base_key, None)
        if outcome is None or pd.isna(outcome):
            print(f"[WARN] Outcome missing for {base_fname}")
            return None
        return outcome
    except Exception as e:
        print(f"[ERROR] Failed to extract outcome from {path}: {e}")
        return None


def filter_dataset_by_outcome(images, y1, y2, y3):
    """
    Filters dataset samples by valid outcomes (0.0 or 1.0).
    
    Returns:
        tuple: (filtered_x, filtered_y1, filtered_y2, filtered_y3, filtered_labels)
    """

    # Load mapping
    if not os.path.exists(MASTERLIST_PATH):
        raise FileNotFoundError(f"Masterlist file not found: {MASTERLIST_PATH}")
    
    df = pd.read_excel(MASTERLIST_PATH)
    if "Outcome" not in df.columns or "File Name" not in df.columns:
        raise ValueError("Excel file missing required columns: 'Outcome' or 'File Name'")
    df = df[df['Outcome'] != 2]  # filter out outcome == 2
    image_to_outcome = dict(zip(df['File Name'], df['Outcome']))
    print(f"[INFO] Loaded {len(image_to_outcome)} image→outcome mappings")

    filtered_x, filtered_y1, filtered_y2, filtered_y3, filtered_labels = [], [], [], [], []
    skipped = 0

    for i, image_path in enumerate(images):
        outcome = get_outcome_from_path(image_path, image_to_outcome)
        if outcome in [0.0, 1.0]:
            filtered_x.append(image_path)
            filtered_y1.append(y1[i])
            filtered_y2.append(y2[i])
            filtered_y3.append(y3[i])
            filtered_labels.append(outcome)
        else:
            skipped += 1

    print(f"[INFO] Filtered dataset: {len(filtered_x)} valid samples, {skipped} skipped")
    return filtered_x, filtered_y1, filtered_y2, filtered_y3, filtered_labels