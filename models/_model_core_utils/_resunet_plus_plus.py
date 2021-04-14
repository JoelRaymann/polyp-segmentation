"""
Scripts to implement the ResUNet++ Model
"""
import tensorflow as tf

import layers


def ResUNetPlusPlus(input_size: tuple, test_mode=False):
    """
    Function to build the ResUNet++ model for polyp segmentation

    Parameters
    ----------
    input_size : tuple
        The input tuple of type (image_width, image_height, n_channel)
    test_mode : bool
        A status flag to build the data for testing mode.

    Returns
    -------
    tf.keras.Model
        A built keras ResUNet++ Model.
    """
    assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))

    image_width, image_height, n_channels = input_size

    inp = tf.keras.layers.Input(shape=(image_width, image_height, n_channels), dtype="float32", name="input_layer")

    # starting conv
    x = layers.Conv2DBN(64, 3, padding="same", activation="relu", name="conv_start")(inp)

    # Residual block 1
    x = layers.ResidualBlock(64, 3, activation="relu", name="rb_1")(x)
    skip1 = x

    x = tf.keras.layers.Conv2D(128,
                               kernel_size=(3, 3),
                               strides=(2, 2),
                               padding="same",
                               activation="linear")(x)
    x = layers.SqueezeExcitationBlock(ratio=1)(x)

    # Residual block 2
    x = layers.ResidualBlock(128, 3, activation="relu", name="rb_2")(x)
    skip2 = x

    x = tf.keras.layers.Conv2D(256,
                               kernel_size=(3, 3),
                               strides=(2, 2),
                               padding="same",
                               activation="linear")(x)
    x = layers.SqueezeExcitationBlock(ratio=2)(x)

    # Residual block 3
    x = layers.ResidualBlock(256, 3, activation="relu", name="rb_3")(x)
    skip3 = x

    x = tf.keras.layers.Conv2D(512,
                               kernel_size=(3, 3),
                               strides=(2, 2),
                               padding="same",
                               activation="linear")(x)
    x = layers.SqueezeExcitationBlock(ratio=4)(x)

    # Residual block 4
    x = layers.ResidualBlock(512, 3, activation="relu", name="rb_4")(x)
    skip4 = x

    x = tf.keras.layers.Conv2D(1024,
                               kernel_size=(3, 3),
                               strides=(2, 2),
                               padding="same",
                               activation="linear")(x)
    x = layers.SqueezeExcitationBlock(ratio=8)(x)

    # Bottleneck ASPP
    x = layers.ASPP(256, [4, 8, 12], (256, 256), 16, activation="relu", name="aspp_bottleneck")(x)
    x = layers.Conv2DBN(1024, 1, activation="relu")(x)

    # Up-sample L4
    x = layers.GlobalAttentionUpsample(name="GAU_4")([skip4, x])
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip4])
    x = tf.keras.layers.Conv2D(512,
                               kernel_size=(3, 3),
                               padding="same",
                               activation="linear")(x)
    x = layers.ResidualBlock(512, 3, activation="relu", activate_begin=True, name="u_rb_4")(x)

    # Up-sample L3
    x = layers.GlobalAttentionUpsample(name="GAU_3")([skip3, x])
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip3])
    x = tf.keras.layers.Conv2D(256,
                               kernel_size=(3, 3),
                               padding="same",
                               activation="linear")(x)
    x = layers.ResidualBlock(256, 3, activation="relu", activate_begin=True, name="u_rb_3")(x)

    # Up-sample L2
    x = layers.GlobalAttentionUpsample(name="GAU_2")([skip2, x])
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip2])
    x = tf.keras.layers.Conv2D(128,
                               kernel_size=(3, 3),
                               padding="same",
                               activation="linear")(x)
    x = layers.ResidualBlock(128, 3, activation="relu", activate_begin=True, name="u_rb_2")(x)

    # Up-sample L1
    x = layers.GlobalAttentionUpsample(name="GAU_1")([skip1, x])
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip1])
    x = tf.keras.layers.Conv2D(64,
                               kernel_size=(3, 3),
                               padding="same",
                               activation="linear")(x)
    x = layers.ResidualBlock(64, 3, activation="relu", activate_begin=True, name="u_rb_1")(x)
    x = layers.Conv2DBN(1, 1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=[inp], outputs=[x])
