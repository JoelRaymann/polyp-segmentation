"""
Script to implement SEUNet Model
"""
import tensorflow as tf

import layers


def SEUNet(input_size: tuple, test_mode=False) -> tf.keras.Model:
    """
    A function that builds and returns SEUNet model

    Parameters
    ----------
    input_size: tuple
        The input size as tuple (width, height, n_channels)
    test_mode: bool, optional
        A status code to initiate the model for testing purpose only. (default: False)

    Returns
    -------
    tf.keras.Model
        The built SEUNet model
    """

    assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))

    image_width, image_height, n_channels = input_size

    inp = tf.keras.layers.Input(shape=(image_width, image_height, n_channels), dtype="float32", name="input_layer")

    # Level-1 conv
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="L1_conv1")(inp)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="L1_conv2")(x)
    skip1 = x
    x = tf.keras.layers.MaxPool2D(name="L1_pool")(x)

    # Level-2 conv
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="L2_conv1")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="L2_conv2")(x)
    skip2 = x
    x = tf.keras.layers.MaxPool2D(name="L2_pool")(x)

    # Level-3 conv
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="L3_conv1")(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="L3_conv2")(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="L3_conv3")(x)
    skip3 = x
    x = tf.keras.layers.MaxPool2D(name="L3_pool")(x)

    # Level-4 conv
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="L4_conv1")(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="L4_conv2")(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="L4_conv3")(x)
    skip4 = x
    x = tf.keras.layers.MaxPool2D(name="L4_pool")(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same", name="BTN_conv1")(x)
    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same", name="BTN_conv2")(x)
    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same", name="BTN_conv3")(x)
    x = layers.ASPP(256,
                    [1, 2, 4],
                    input_size=(256, 256),
                    scale_rate=16,
                    activation="relu",
                    name="BTN_ASPP")(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="BTN_ASPP_Projection")(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="BTN_conv4")(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="BTN_conv5")(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="BTN_conv6")(x)

    # UP Level-4
    x = tf.keras.layers.UpSampling2D(interpolation="bilinear", name="U4_up")(x)
    x = tf.keras.layers.Concatenate(axis=-1, name="U4_concat")([skip4, x])
    x = layers.SqueezeExcitationBlock(ratio=16, name="U4_SE")(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="U4_conv1")(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="U4_conv2")(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="U4_conv3")(x)

    # UP Level-3
    x = tf.keras.layers.UpSampling2D(interpolation="bilinear", name="U3_up")(x)
    x = tf.keras.layers.Concatenate(axis=-1, name="U3_concat")([skip3, x])
    x = layers.SqueezeExcitationBlock(ratio=8, name="U3_SE")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="U3_conv1")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="U3_conv2")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="U3_conv3")(x)

    # UP Level-2
    x = tf.keras.layers.UpSampling2D(interpolation="bilinear", name="U2_up")(x)
    x = tf.keras.layers.Concatenate(axis=-1, name="U2_concat")([skip2, x])
    x = layers.SqueezeExcitationBlock(ratio=4, name="U2_SE")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="U2_conv1")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="U2_conv2")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="U2_conv3")(x)

    # UP Level-2
    x = tf.keras.layers.UpSampling2D(interpolation="bilinear", name="U1_up")(x)
    x = tf.keras.layers.Concatenate(axis=-1, name="U1_concat")([skip1, x])
    x = layers.SqueezeExcitationBlock(ratio=2, name="U1_SE")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="U1_conv1")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="U1_conv2")(x)
    out = tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same", name="output_layer")(x)

    return tf.keras.Model(inputs=[inp], outputs=[out])


