import tensorflow as tf

SMOOTH = 1e-15


def iou_multiclass(y_true, y_pred):
    y_true = tf.round(y_true)
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0,1,2])  # sum per channel
    union = tf.reduce_sum(y_true, axis=[0,1,2]) + tf.reduce_sum(y_pred, axis=[0,1,2]) - intersection
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return tf.reduce_mean(iou)

def dice_coef_multiclass(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0,1,2])
    denominator = tf.reduce_sum(y_true, axis=[0,1,2]) + tf.reduce_sum(y_pred, axis=[0,1,2])
    dice = (2. * intersection + SMOOTH) / (denominator + SMOOTH)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef_multiclass(y_true, y_pred)