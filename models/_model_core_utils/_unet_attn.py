"""
Script to implement the U-Net Attention Architecture for training/testing
"""

# Import the necessary packages
import tensorflow as tf

# Import in-house packages
import layers


def _conv_block(X, filters: int, kernel_size: int, activation: str, name: str, batch_norm=True):

    conv1 = tf.keras.layers.Conv2D(filters, kernel_size, activation=None, padding="same", name="conv1_" + name)(X)
    if batch_norm:
        conv1 = tf.keras.layers.BatchNormalization(axis=-1)(conv1)
    conv1 = tf.keras.layers.Activation(activation)(conv1)

    conv2 = tf.keras.layers.Conv2D(filters, kernel_size, activation=None, padding="same", name="conv2_" + name)(conv1)
    if batch_norm:
        conv2 = tf.keras.layers.BatchNormalization(axis=-1)(conv2)
    conv2 = tf.keras.layers.Activation(activation)(conv2)

    return conv2


def UNetAttn(input_size: tuple, test_mode=False):
    """
    Function to build the U-Net Attention architecture adapted for Dice index training.

    Parameters
    ----------
    input_size : tuple
        The input tuple size to build the model. (image_width, image_height, n_channel)
    test_mode : bool, optional
        The status flag to request building the model for deployment.

    Returns
    -------
    tf.keras.Model
        The keras model of the U-Net Attention architecture
    """
    # Get the inputs
    assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))

    image_width, image_height, n_channels = input_size

    # Developing model here
    X = tf.keras.layers.Input(shape=(image_width, image_width, n_channels), dtype="float32", name="input_layer")

    # Level - 1 Down
    L1_conv = _conv_block(X, 64, 3, "elu", name="L1", batch_norm=True)
    L1_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_L1")(L1_conv)

    # Level - 2 Down
    L2_conv = _conv_block(L1_pool, 128, 3, "elu", name="L2", batch_norm=True)
    L2_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_L2")(L2_conv)

    # Level - 3 Down
    L3_conv = _conv_block(L2_pool, 256, 3, "elu", name="L3", batch_norm=True)
    L3_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_L3")(L3_conv)

    # Level - 4 Down
    L4_conv = _conv_block(L3_pool, 512, 3, "elu", name="L4", batch_norm=True)
    L4_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_L4")(L4_conv)

    # Bridge
    B_conv = _conv_block(L4_pool, 1024, 3, "elu", name="Bridge", batch_norm=True)
    drop = tf.keras.layers.SpatialDropout2D(0.3)(B_conv)

    # Level - 4 up
    U4_up = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), activation="elu", name="U4_up")(drop)
    U4_attn, U4_map = layers.AttentionLayer(512, name="U4_attn")(L4_conv, U4_up)
    U4_merge = tf.keras.layers.Concatenate(axis=-1, name="U4_merge")([U4_up, U4_attn])
    U4_conv = _conv_block(U4_merge, 512, 3, "elu", name="U4", batch_norm=True)

    # Level - 3 up
    U3_up = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), activation="elu", name="U3_up")(U4_conv)
    U3_attn, U3_map = layers.AttentionLayer(256, name="U3_attn")(L3_conv, U3_up)
    U3_merge = tf.keras.layers.Concatenate(axis=-1, name="U3_merge")([U3_up, U3_attn])
    U3_conv = _conv_block(U3_merge, 256, 3, "elu", name="U3", batch_norm=True)

    # Level - 2 up
    U2_up = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), activation="elu", name="U2_up")(U3_conv)
    U2_attn, U2_map = layers.AttentionLayer(128, name="U2_attn")(L2_conv, U2_up)
    U2_merge = tf.keras.layers.Concatenate(axis=-1, name="U2_merge")([U2_up, U2_attn])
    U2_conv = _conv_block(U2_merge, 128, 3, "elu", name="U2", batch_norm=True)

    # Level - 1 up
    U1_up = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), activation="elu", name="U1_up")(U2_conv)
    U1_attn, U1_map = layers.AttentionLayer(64, name="U1_attn")(L1_conv, U1_up)
    U1_merge = tf.keras.layers.Concatenate(axis=-1, name="U1_merge")([U1_up, U1_attn])
    U1_conv = _conv_block(U1_merge, 64, 3, "elu", name="U1", batch_norm=True)

    # Output Layer
    y = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", padding="same", name="output_layer")(U1_conv)

    return tf.keras.Model(inputs=X, outputs=[y, U1_map, U2_map, U3_map, U4_map])
