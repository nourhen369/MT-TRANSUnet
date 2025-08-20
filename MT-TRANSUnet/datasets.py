import pandas as pd
import numpy as np
import tensorflow as tf
import os, cv2, re, random
from collections import defaultdict, Counter
from dotenv import load_dotenv

load_dotenv()
SIZE = os.getenv("SIZE", 256)

def get_outcome_from_path(
        path,
        masterlist_path = '/content/drive/MyDrive/MasterlistAug30-2017.xlsx'
    ):
    '''
    Extracts the outcome from the file path based on a masterlist Excel file.

    Args:
        path (str): The file path from which to extract the outcome.
        masterlist_path (str): Path to the masterlist Excel file.
    '''
    fname = os.path.basename(path)
    base_fname = os.path.splitext(fname)[0]
    base_key = re.sub(r'_\d+$', '', base_fname)

    df = pd.read_excel(masterlist_path)
    df = df[df['Outcome'] != 2]
    image_to_outcome = dict(zip(df['File Name'], df['Outcome']))
    
    outcome_counts = df['Outcome'].value_counts()
    print("Outcome distribution:\n", outcome_counts)

    outcome = image_to_outcome.get(base_key, None)
    if outcome is None or pd.isna(outcome):
        pass
    return outcome

def read_image_tf(path):
    '''
    Extracts the embryo image from the file path.

    Args:
        path (str): The file path from which to extract the bmp image.
    '''
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (SIZE, SIZE))
    x = x / 255.0
    return x.astype(np.float32)

def read_mask_tf(path):
    '''
    Extracts the embryo mask (ICM, ZP, TE) from the file path.

    Args:
        path (str): The file path from which to extract the bmp mask.
    '''
    path = path.decode()
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (SIZE, SIZE))
    y[y != 0] = 255
    y = y / 255.0
    y = (y > 0.5).astype(np.float32)
    y = np.expand_dims(y, axis=-1)
    return y

def read_multiclass_mask_tf(te_path, zp_path, icm_path):
    '''
    Concatenates the embryo masks (ICM, ZP, TE) into one single mask from the masks file path.

    Args:
        te_path (str): Path to TE mask file.
        zp_path (str): Path to ZP mask file.
        icm_path (str): Path to ICM mask file.
    '''
    te = read_mask_tf(te_path)
    zp = read_mask_tf(zp_path)
    icm = read_mask_tf(icm_path)
    mask = np.concatenate([te, zp, icm], axis=-1)  # (H, W, 3)
    return mask

def tf_parse_multi_with_cls(x_path, te_path, zp_path, icm_path, cls_label):
    """
    Parses input image and corresponding multi-class masks along with a classification label.
    Args:
        x_path (tf.Tensor): Path to the input image file.
        te_path (tf.Tensor): Path to the TE mask file.
        zp_path (tf.Tensor): Path to the ZP mask file.
        icm_path (tf.Tensor): Path to the ICM mask file.
        cls_label (tf.Tensor): Classification label associated with the image.
    Returns:
        A tuple (x, (y, cls_label)) where:
            - x (tf.Tensor): The parsed image tensor of shape [SIZE, SIZE, 3] and dtype tf.float32.
            - y (tf.Tensor): The stacked multi-class mask tensor of shape [SIZE, SIZE, 3] and dtype tf.float32, with channels corresponding to TE, ZP, and ICM.
            - cls_label (tf.Tensor): The classification label tensor of shape [] and dtype tf.int32.
    """
    def _parse(x_path, te_path, zp_path, icm_path, cls_label):
        x = read_image_tf(x_path)
        y = read_multiclass_mask_tf(te_path, zp_path, icm_path)
        return x, y, cls_label

    x, y, cls_label = tf.numpy_function(
        _parse,
        [x_path, te_path, zp_path, icm_path, cls_label],
        [tf.float32, tf.float32, tf.int32]
    )
    x.set_shape([SIZE, SIZE, 3])
    y.set_shape([SIZE, SIZE, 3])  # 3 channels: TE, ZP, ICM
    cls_label.set_shape([])
    return x, (y, cls_label)

def tf_dataset_multi_with_cls(x_paths, te_paths, zp_paths, icm_paths, cls_labels, batch_size=16, shuffle=True):
    """
    Creates a TensorFlow dataset for multi-input data with classification labels, from list or tensor of image/mask file paths.
    Returns:
        A TensorFlow dataset yielding batches of parsed and preprocessed data, ready for model training or evaluation.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_paths, te_paths, zp_paths, icm_paths, cls_labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(tf_parse_multi_with_cls, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def filter_dataset_by_outcome(images, y1, y2, y3):
    """
    Filters the dataset based on the outcome extracted from image paths.
    This function iterates over the provided list of image paths and selects only those
    whose outcome is either 0 (Not implanted) or 1 (Implanted).
    """
    filtered_x = []
    filtered_y1 = []
    filtered_y2 = []
    filtered_y3 = []
    filtered_labels = []

    for i, image_path in enumerate(images):
        outcome = get_outcome_from_path(image_path)
        if outcome in [0.0, 1.0]:
            filtered_x.append(image_path)
            filtered_y1.append(y1[i])
            filtered_y2.append(y2[i])
            filtered_y3.append(y3[i])
            filtered_labels.append(int(outcome))
    return filtered_x, filtered_y1, filtered_y2, filtered_y3, filtered_labels

def handle_class_imbalance(labels, x, y1, y2, y3):
    '''
    Undersamples majority class.
    '''
    print("Before undersampling:", Counter(labels))

    indices_by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_class[label].append(idx)

    minority_class_size = min(len(indices_by_class[0]), len(indices_by_class[1]))
    undersampled_indices = indices_by_class[0] + random.sample(indices_by_class[1], minority_class_size)
    random.shuffle(undersampled_indices)

    undersampled_x = [x[i] for i in undersampled_indices]
    undersampled_y1 = [y1[i] for i in undersampled_indices]
    undersampled_y2 = [y2[i] for i in undersampled_indices]
    undersampled_y3 = [y3[i] for i in undersampled_indices]
    undersampled_labels = [labels[i] for i in undersampled_indices]
    print("After undersampling:", Counter(undersampled_labels))

    return tf_dataset_multi_with_cls(
        undersampled_x,
        undersampled_y1,
        undersampled_y2,
        undersampled_y3,
        undersampled_labels,
        batch_size=8,
    )