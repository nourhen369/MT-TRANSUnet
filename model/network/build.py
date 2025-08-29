import yaml
import tensorflow as tf
from tensorflow.keras import layers, Model

from model.network.tranformer import *
from model.network.cls_head import classification_head_combined


with open("model/config.yaml") as f:
    config = yaml.safe_load(f)

SIZE = config["hyperparams"]["size"]
INPUT_SHAPE = (SIZE, SIZE, 3)

def build_transunet(return_attention=False):
    inputs = tf.keras.Input(INPUT_SHAPE)

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

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    seg_output = layers.Conv2D(3, 1, padding="same", activation="sigmoid", name="segmentation")(d4)
    cls_output = classification_head_combined(p3, b1)

    if return_attention:
        return Model(inputs, [seg_output, cls_output, attention_maps])
    return Model(inputs, [seg_output, cls_output])