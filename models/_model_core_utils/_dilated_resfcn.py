"""
Script to implement dilated ResFCN model
"""
import tensorflow as tf


def DilatedResFCN(input_size: tuple, test_mode=False) -> tf.keras.Model:
    """
    A function that builds and returns Dilated ResFCN model

    Parameters
    ----------
    input_size: tuple
        The input size as tuple (width, height, n_channels)
    test_mode: bool, optional
        A status code to initiate the model for testing purpose only. (default: False)

    Returns
    -------
    tf.keras.Model
        The built Dilated ResFCN model
    """

    assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))

    image_width, image_height, n_channels = input_size

    inp = tf.keras.layers.Input(shape=(image_width, image_height, n_channels), dtype="float32", name="input_layer")

    # Get the resnet50 model
    resnet50 = tf.keras.applications.ResNet50(weights="imagenet",
                                              pooling=None,
                                              input_shape=(image_width, image_height, n_channels),
                                              include_top=False)
    resnet50_extracted = tf.keras.Model(inputs=[resnet50.inputs], outputs=[
        resnet50.get_layer("conv2_block3_out").output,
        resnet50.get_layer("conv3_block4_out").output,
        resnet50.get_layer("conv4_block6_out").output,
        resnet50.get_layer("conv5_block3_out").output
    ])
    res2_out, res3_out, res4_out, res5_out = resnet50_extracted(inp)

    # Build res5
    res5_out = tf.keras.layers.Conv2D(2048, 3, activation="relu", padding="same")(res5_out)
    res5_out = tf.keras.layers.Conv2D(256, 1, activation="relu", padding="same")(res5_out)
    res5_out = tf.keras.layers.Dropout(0.5)(res5_out)
    res5_out = tf.keras.layers.Conv2D(2, 1, activation="relu", padding="same")(res5_out)
    res5_out = tf.keras.layers.Conv2DTranspose(2, 2, strides=(2, 2), activation="relu")(res5_out)

    # Build res4
    res4_out = tf.keras.layers.Conv2D(1024, 3, dilation_rate=4, activation="relu", padding="same")(res4_out)
    res4_out = tf.keras.layers.Conv2D(256, 1, activation="relu", padding="same")(res4_out)
    res4_out = tf.keras.layers.Dropout(0.5)(res4_out)
    res4_out = tf.keras.layers.Conv2D(2, 1, activation="relu", padding="same")(res4_out)
    res4_out = tf.keras.layers.Add()([res4_out, res5_out])
    res4_out = tf.keras.layers.Conv2DTranspose(2, 2, strides=(2, 2), activation="relu")(res4_out)

    # Build res3
    res3_out = tf.keras.layers.Conv2D(512, 3, dilation_rate=8, activation="relu", padding="same")(res3_out)
    res3_out = tf.keras.layers.Conv2D(256, 1, activation="relu", padding="same")(res3_out)
    res3_out = tf.keras.layers.Dropout(0.5)(res3_out)
    res3_out = tf.keras.layers.Conv2D(2, 1, activation="relu", padding="same")(res3_out)
    res3_out = tf.keras.layers.Add()([res3_out, res4_out])
    res3_out = tf.keras.layers.Conv2DTranspose(2, 2, strides=(2, 2), activation="relu")(res3_out)

    # Build res3
    res2_out = tf.keras.layers.Conv2D(256, 3, dilation_rate=16, activation="relu", padding="same")(res2_out)
    res2_out = tf.keras.layers.Conv2D(256, 1, activation="relu", padding="same")(res2_out)
    res2_out = tf.keras.layers.Dropout(0.5)(res2_out)
    res2_out = tf.keras.layers.Conv2D(2, 1, activation="relu", padding="same")(res2_out)
    res2_out = tf.keras.layers.Add()([res2_out, res3_out])
    res2_out = tf.keras.layers.Conv2DTranspose(2, 2, strides=(2, 2), activation="relu")(res2_out)

    out = tf.keras.layers.Conv2DTranspose(2, 2, strides=(2, 2), activation="softmax")(res2_out)

    return tf.keras.Model(inputs=[inp], outputs=[out])


