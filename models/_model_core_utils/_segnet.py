"""
Script to implement the SegNet model
"""
# import necessary packages
import tensorflow as tf

# import in-house packages
import layers


def _conv_bn(x, filters: int, kernel_size: int, activation: str, name: str):

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="linear", name="{0}_conv".format(name)),
        tf.keras.layers.BatchNormalization(axis=-1, name="{0}_conv_BN".format(name)),
        tf.keras.layers.Activation(activation=activation, name="{0}_conv_act".format(name))
    ], name="{0}_conv_with_BN".format(name))(x)


def SegNet(input_size:tuple, test_mode=False) -> tf.keras.Model:
    """
    Function to build the SegNet model.

    Parameters
    ----------
    input_size: tuple
        The input_size tuple of (width, height, n_channels)
    test_mode: bool, optional
        The status mode to build the model for test mode or training

    Returns
    -------
    tf.keras.Model
        The keras SegNet model
    """

    # Get the inputs
    assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))

    image_width, image_height, n_channels = input_size

    # Developing model here
    inp = tf.keras.layers.Input(shape=(image_width, image_width, n_channels), dtype="float32", name="input_layer")

    # Level - 1 Down
    x = _conv_bn(inp, 64, 3, activation="relu", name="L1_D_conv1")
    x = _conv_bn(x, 64, 3, activation="relu", name="L1_D_conv2")
    x, mask_1 = layers.MaxPoolingWithIndicing2D(pool_size=(2, 2), padding="same", name="L1_D_pool1")(x)

    # Level - 2 Down
    x = _conv_bn(x, 128, 3, activation="relu", name="L2_D_conv1")
    x = _conv_bn(x, 128, 3, activation="relu", name="L2_D_conv2")
    x, mask_2 = layers.MaxPoolingWithIndicing2D(pool_size=(2, 2), padding="same", name="L2_D_pool1")(x)

    # Level - 3 Down
    x = _conv_bn(x, 256, 3, activation="relu", name="L3_D_conv1")
    x = _conv_bn(x, 256, 3, activation="relu", name="L3_D_conv2")
    x = _conv_bn(x, 256, 3, activation="relu", name="L3_D_conv3")
    x, mask_3 = layers.MaxPoolingWithIndicing2D(pool_size=(2, 2), padding="same", name="L3_D_pool1")(x)

    # Level - 4 Down
    x = _conv_bn(x, 512, 3, activation="relu", name="L4_D_conv1")
    x = _conv_bn(x, 512, 3, activation="relu", name="L4_D_conv2")
    x = _conv_bn(x, 512, 3, activation="relu", name="L4_D_conv3")
    x, mask_4 = layers.MaxPoolingWithIndicing2D(pool_size=(2, 2), padding="same", name="L4_D_pool1")(x)

    # Level - 5 Down
    x = _conv_bn(x, 512, 3, activation="relu", name="L5_D_conv1")
    x = _conv_bn(x, 512, 3, activation="relu", name="L5_D_conv2")
    x = _conv_bn(x, 512, 3, activation="relu", name="L5_D_conv3")
    x, mask_5 = layers.MaxPoolingWithIndicing2D(pool_size=(2, 2), padding="same", name="L5_D_pool1")(x)

    # Level - 5 Up
    x = layers.MaxUnpooling2D(size=(2, 2), name="L5_U_unpool1")([x, mask_5])
    x = _conv_bn(x, 512, 3, activation="relu", name="L5_U_conv1")
    x = _conv_bn(x, 512, 3, activation="relu", name="L5_U_conv2")
    x = _conv_bn(x, 512, 3, activation="relu", name="L5_U_conv3")

    # Level - 4 Up
    x = layers.MaxUnpooling2D(size=(2, 2), name="L4_U_unpool1")([x, mask_4])
    x = _conv_bn(x, 512, 3, activation="relu", name="L4_U_conv1")
    x = _conv_bn(x, 512, 3, activation="relu", name="L4_U_conv2")
    x = _conv_bn(x, 256, 3, activation="relu", name="L4_U_conv3")

    # Level - 3 Up
    x = layers.MaxUnpooling2D(size=(2, 2), name="L3_U_unpool1")([x, mask_3])
    x = _conv_bn(x, 256, 3, activation="relu", name="L3_U_conv1")
    x = _conv_bn(x, 256, 3, activation="relu", name="L3_U_conv2")
    x = _conv_bn(x, 128, 3, activation="relu", name="L3_U_conv3")

    # Level - 2 Up
    x = layers.MaxUnpooling2D(size=(2, 2), name="L2_U_unpool1")([x, mask_2])
    x = _conv_bn(x, 128, 3, activation="relu", name="L2_U_conv1")
    x = _conv_bn(x, 64, 3, activation="relu", name="L2_U_conv2")

    # Level - 1 Up
    x = layers.MaxUnpooling2D(size=(2, 2), name="L1_U_unpool1")([x, mask_1])
    x = _conv_bn(x, 64, 3, activation="relu", name="L1_U_conv1")

    # output layer
    y = _conv_bn(x, 1, 1, activation="sigmoid", name="output_layer")

    return tf.keras.Model(inputs=inp, outputs=y)
