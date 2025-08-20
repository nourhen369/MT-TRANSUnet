import tensorflow as tf


smooth = 1e-15

def binary_focal_loss(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = alpha_t * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(loss)
    return focal_loss

def iou_multiclass(y_true, y_pred):
    y_true = tf.round(y_true)
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0,1,2])  # sum per channel
    union = tf.reduce_sum(y_true, axis=[0,1,2]) + tf.reduce_sum(y_pred, axis=[0,1,2]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

def dice_coef_multiclass(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0,1,2])
    denominator = tf.reduce_sum(y_true, axis=[0,1,2]) + tf.reduce_sum(y_pred, axis=[0,1,2])
    dice = (2. * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef_multiclass(y_true, y_pred)

class GradualLossWeight(tf.keras.callbacks.Callback):
    def __init__(self, start_weight=0.5, end_weight=1.0, ramp_epochs=10):
        super().__init__()
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.ramp_epochs = ramp_epochs

    def on_epoch_begin(self, epoch):
        if epoch < self.ramp_epochs:
            new_weight = self.start_weight + (self.end_weight - self.start_weight) * (epoch / self.ramp_epochs)
        else:
            new_weight = self.end_weight

        self.model.loss_weights = {"segmentation": 1.0, "classification": new_weight}
        print(f"Epoch {epoch+1}: classification loss weight = {new_weight:.4f}")