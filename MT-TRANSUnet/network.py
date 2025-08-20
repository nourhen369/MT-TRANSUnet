import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import os
from dotenv import load_dotenv


load_dotenv()
SIZE = os.getenv("SIZE", 256)
input_shape = (SIZE, SIZE, 3)
weight_decay=1e-8


def conv_block(x, filters, dropout=False):
    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout:
        x = layers.Dropout(0.5)(x)

    return x

def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def classification_head(bottleneck_feat, encoder_mid_feat):
    cls_feat = layers.Conv2D(256, 3, padding='same', activation='swish')(bottleneck_feat)
    cls_feat = layers.BatchNormalization()(cls_feat)
    cls_feat = layers.Dropout(0.3)(cls_feat)

    s3_resized = layers.Conv2D(256, 1, padding='same')(encoder_mid_feat)
    s3_resized = layers.BatchNormalization()(s3_resized)
    if s3_resized.shape[1:3] != cls_feat.shape[1:3]:
        s3_resized = layers.Resizing(cls_feat.shape[1], cls_feat.shape[2])(s3_resized)

    fused_feat = layers.Concatenate(axis=-1)([cls_feat, s3_resized])

    gap = layers.GlobalAveragePooling2D()(fused_feat)
    gmp = layers.GlobalMaxPooling2D()(fused_feat)
    x = layers.Concatenate()([gap, gmp])

    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x_res = layers.Dense(256, activation='swish')(x)
    x = layers.Add()([x, x_res])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='swish')(x)
    x = layers.Dropout(0.3)(x)

    return layers.Dense(1, activation='sigmoid', name="classification")(x)

# image sequentialization: shaping the input x into a sequence of flattened 2D patches
def patch_embedding(x, patch_size, embed_dim):
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(x)
    # (B, H/P, W/P, D=embed_dim), reshaped to (B, N, D), N=H*W/P²
    x = layers.Reshape((-1, embed_dim))(x)
    return x

def transformer_block(x, embed_dim, num_heads, mlp_dim, dropout=0.1, return_attention=False):
    norm1 = layers.LayerNormalization(epsilon=1e-6)(x)
    mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)

    if return_attention:
        attn_output, attn_scores = mha(norm1, norm1, return_attention_scores=True)
    else:
        attn_output = mha(norm1, norm1)

    x = layers.Add()([x, attn_output])

    norm2 = layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_output = layers.Dense(mlp_dim, activation='gelu')(norm2)
    mlp_output = layers.Dropout(dropout)(mlp_output)
    mlp_output = layers.Dense(embed_dim)(mlp_output)
    mlp_output = layers.Dropout(dropout)(mlp_output)
    x = layers.Add()([x, mlp_output])

    if return_attention:
        return x, attn_scores
    return x

def transformer_bottleneck(x, patch_size=1, num_layers=4, embed_dim=512, num_heads=8, mlp_dim=1024, return_all_attention=False):
    h, w, c = x.shape[1:]

    # Image Sequentialization.
    x = patch_embedding(x, patch_size, embed_dim)  # (B, N, D)

    # Positional encoding
    pos_emb = tf.Variable(tf.random.normal([1, x.shape[1], embed_dim]), trainable=True)
    x = x + pos_emb # Patch Embedding.

    attn_maps = []  # attention scores
    for _ in range(num_layers):
        if return_all_attention:
            x, attn = transformer_block(x, embed_dim, num_heads, mlp_dim, return_attention=True)
            attn_maps.append(attn)
        else:
            x = transformer_block(x, embed_dim, num_heads, mlp_dim)

    # Reshape back to 2D
    new_h, new_w = h // patch_size, w // patch_size
    x = layers.Reshape((new_h, new_w, embed_dim))(x)

    if return_all_attention:
        return x, attn_maps
    return x

def build_transunet(input_shape, return_attention=False):
    inputs = tf.keras.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4 = conv_block(p3, 512, dropout=True)
    p4 = layers.MaxPooling2D((2, 2))(s4)

    # Transformer bottleneck
    if return_attention:
        b1, attention_maps = transformer_bottleneck(p4, patch_size=1, num_layers=4,
                                                    embed_dim=512, num_heads=8,
                                                    mlp_dim=1024, return_all_attention=True)
    else:
        b1 = transformer_bottleneck(p4, patch_size=1, num_layers=4,
                                    embed_dim=512, num_heads=8,
                                    mlp_dim=1024, return_all_attention=False)

    # Decoder for segmentation
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    seg_output = layers.Conv2D(3, 1, padding="same", activation="sigmoid", name="segmentation")(d4)
    cls_output = classification_head(b1, s3)

    if return_attention:
        return Model(inputs, [seg_output, cls_output, attention_maps])
    return Model(inputs, [seg_output, cls_output])