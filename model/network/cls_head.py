from tensorflow.keras import layers, regularizers

"""
    This file contains various classification heads tested in our model.
    Finally selected classification_head_combined
"""


def classification_head_v1(bottleneck_feat):
  gap = layers.GlobalAveragePooling1D()(bottleneck_feat)
  x = layers.Dense(128, activation='relu')(gap)
  x = layers.Dropout(0.4)(x)
  return layers.Dense(1, activation='sigmoid', name="classification")(x)

def classification_head_v2(bottleneck_feat, dropout_rate=0.3, l2_reg=1e-4):
    x = layers.GlobalAveragePooling2D()(bottleneck_feat)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1, activation='sigmoid', name="classification")(x)
    return output

def cls_head_midfeature(mid_feat):
    x = layers.GlobalAveragePooling2D()(mid_feat)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    cls_output = layers.Dense(1, activation='sigmoid', name='classification')(x)
    return cls_output

def classification_head_combined(mid_feat, bottleneck_feat, dropout_rate=0.5, l2_reg=1e-4):
    mid_pool = layers.GlobalAveragePooling2D()(mid_feat)
    b1_reshaped = layers.Reshape((-1, bottleneck_feat.shape[-1]))(bottleneck_feat)  # (B, N_patches, D)
    bottleneck_pool = layers.GlobalAveragePooling1D()(b1_reshaped)  # shape -> (B, D)

    x = layers.Concatenate()([mid_pool, bottleneck_pool])  # shape -> (B, C_mid + D)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    return layers.Dense(1, activation='sigmoid', name="classification")(x)