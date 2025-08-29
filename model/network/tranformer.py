import tensorflow as tf
from tensorflow.keras import layers


def conv_block(x, filters, dropout=False):
    """
    Creates an UNet convolutional block with two Conv2D layers, BatchNorm, and ReLU activations.
    """
    try:
        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        if dropout:
            x = layers.Dropout(0.5)(x)

        return x
    except Exception as e:
        print(f"[ERROR] conv_block failed with filters={filters}: {e}")
        return x


def encoder_block(x, filters):
    """
    Encoder block with convolution and downsampling.

    Returns:
        tuple: (conv_output, pooled_output)
    """
    try:
        c = conv_block(x, filters)
        p = layers.MaxPooling2D((2, 2))(c)
        return c, p
    except Exception as e:
        print(f"[ERROR] encoder_block failed with filters={filters}: {e}")
        return x, x


def decoder_block(x, skip, filters):
    """
    Decoder block with upsampling, skip connection, and conv block.
    """
    try:
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x
    except Exception as e:
        print(f"[ERROR] decoder_block failed with filters={filters}: {e}")
        return x


def patch_embedding(x, patch_size, embed_dim):
    """
    Image Sequentialization: Splits input image into patches and embeds them into a sequence.
    
    Args:
        x (tf.Tensor): Input tensor (B, H, W, D).
        patch_size (int): Size of each patch.
        embed_dim (int): Embedding dimension.
    
    Returns:
        tf.Tensor: Patch embeddings (B, N, D).
    """
    try:
        x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(x)
        # (B, H/P, W/P, D=embed_dim), reshaped to (B, N, D), N=H*W/P²
        x = layers.Reshape((-1, embed_dim))(x)
        return x
    except Exception as e:
        print(f"[ERROR] patch_embedding failed with patch_size={patch_size}, embed_dim={embed_dim}: {e}")
        return x


def transformer_block(x, embed_dim, num_heads, mlp_dim, dropout=0.1, return_attention=False):
    """
    Transformer block with multi-head attention and MLP.
    """
    try:
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
    except Exception as e:
        print(f"[ERROR] transformer_block failed: embed_dim={embed_dim}, num_heads={num_heads}, mlp_dim={mlp_dim}: {e}")
        return x


def transformer_bottleneck(x, patch_size=1, num_layers=4, embed_dim=512, num_heads=8, mlp_dim=1024, return_all_attention=False):
    """
    Transformer bottleneck for TRANS-UNet architecture.
    
    Args:
        x (tf.Tensor): Input tensor (B, H, W, C).
        patch_size (int): Size of patch embedding.
        num_layers (int): Number of transformer layers.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension of MLP.
        return_all_attention (bool): Whether to return all attention maps.
    
    Returns:
        tf.Tensor or (tf.Tensor, list[tf.Tensor]): Output tensor, optionally attention maps.
    """
    try:
        h, w, c = x.shape[1:]

        # Patch embedding
        x = patch_embedding(x, patch_size, embed_dim)

        # Positional encoding
        pos_emb = tf.Variable(tf.random.normal([1, x.shape[1], embed_dim]), trainable=True)
        x = x + pos_emb

        attn_maps = []
        for _ in range(num_layers):
            if return_all_attention:
                x, attn = transformer_block(x, embed_dim, num_heads, mlp_dim, return_attention=True)
                attn_maps.append(attn)
            else:
                x = transformer_block(x, embed_dim, num_heads, mlp_dim)

        # Reshape back to 2D feature map
        new_h, new_w = h // patch_size, w // patch_size
        x = layers.Reshape((new_h, new_w, embed_dim))(x)

        if return_all_attention:
            return x, attn_maps
        return x
    except Exception as e:
        print(f"[ERROR] transformer_bottleneck failed: {e}")
        return x