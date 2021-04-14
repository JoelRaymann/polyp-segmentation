"""
Scripts to build the AI-Net
"""
# import necessary packages
import tensorflow as tf

# import in-house packages
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


def _seperable_conv(X, filters: int, kernel_size: int, stride: int, dilation_rate: int, activation: str,
                    prefix: str, depth_activation=False, epsilon=1e-3):

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        X = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(X)
        depth_padding = 'valid'

    if not depth_activation:
        return tf.keras.Sequential([
            tf.keras.layers.Activation(activation),
            tf.keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride),
                                            dilation_rate=dilation_rate, padding=depth_padding, use_bias=False),
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=epsilon),
            tf.keras.layers.Conv2D(filters, (1, 1), padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=epsilon)
        ], name="{0}_sep_conv".format(prefix))(X)

    elif depth_activation:
        return tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride),
                                            dilation_rate=dilation_rate, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=epsilon),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv2D(filters, (1, 1), padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=epsilon),
            tf.keras.layers.Activation(activation),
        ], name="{0}_sep_conv_depthwise".format(prefix))(X)

    else:
        raise NotImplementedError


def _xception_block(inputs, depth_list: list, prefix: str, skip_conn_type: str, stride: int, dilation_rate: int,
                    activation: str, depth_activation=False, return_skip=False):

    residual = inputs
    for i in range(len(depth_list)):
        residual = _seperable_conv(residual,
                                   depth_list[i],
                                   kernel_size=3,
                                   stride=stride if i == len(depth_list) - 1 else 1,
                                   dilation_rate=dilation_rate,
                                   activation=activation,
                                   prefix="{0}_conv_{1}".format(prefix, i + 1),
                                   depth_activation=depth_activation)
        if i == len(depth_list) - 2:
            skip = residual
    if skip_conn_type == 'conv':
        shortcut = tf.keras.layers.Conv2D(depth_list[-1], kernel_size=1, strides=stride, padding="same",
                                          name="{0}_shortcut_conv".format(prefix), activation="linear")(inputs)
        shortcut = tf.keras.layers.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = tf.keras.layers.Add()([residual, shortcut])
    elif skip_conn_type == 'sum':
        outputs = tf.keras.layers.Add()([residual, inputs])
    elif skip_conn_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def _modified_residual_block(X, depth_list: list, prefix: str, skip_conn_type: str, stride: int, activation="elu",
                             return_skip=False):

    residual = X
    skip = None
    for i in range(len(depth_list)):
        residual = tf.keras.Sequential([
            tf.keras.layers.Conv2D(depth_list[i], 3,
                                   stride if i == len(depth_list) - 1 else 1,
                                   padding="same", activation="linear"),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Activation(activation)
        ], name="{0}_sep_conv_{1}".format(prefix, i+1))(residual)

        if i == len(depth_list) - 2:
            skip = residual

    if skip_conn_type == "conv":
        shortcut = tf.keras.Sequential([
            tf.keras.layers.Conv2D(depth_list[-1], 3, stride, padding="same", activation="linear"),
            tf.keras.layers.BatchNormalization(axis=-1),
        ], name="{0}_conv_skip".format(prefix))(X)
        outputs = tf.keras.layers.Add()([residual, shortcut])

    elif skip_conn_type == "sum":
        outputs = tf.keras.layers.Add()([residual, X])

    elif skip_conn_type == "none":
        outputs = residual

    else:
        raise NotImplementedError

    if return_skip:
        return tf.keras.layers.Activation(activation)(outputs), skip
    else:
        return tf.keras.layers.Activation(activation)(outputs)


def GARNet(input_size: tuple, test_mode=False):
    """
    Function to build the Guided Attention modified Residual Network for polyp segmentation.

    Parameters
    ----------
    input_size : tuple
        The input tuple of type (image_width, image_height, n_channel)
    test_mode : bool
        A status flag to build the data for testing mode.

    Returns
    -------
    tf.keras.Model
        A built keras GARNet Model.
    """
    assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))

    image_width, image_height, n_channels = input_size

    # Developing model here
    inp = tf.keras.layers.Input(shape=(image_width, image_width, n_channels), dtype="float32", name="input_layer")

    # Level - 1 down
    x = _conv_block(inp, 64, 3, "elu", name="L1", batch_norm=True)
    skip1 = x
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_L1")(x)

    # Level - 2 Xception down
    x, skip2 = _xception_block(x, [128, 128, 128], "L2_xception", skip_conn_type="conv",
                               stride=2, activation="elu", dilation_rate=1, depth_activation=False,
                               return_skip=True)

    # Level - 3 Xception down
    x, skip3 = _xception_block(x, [256, 256, 256], "L3", skip_conn_type="conv",
                               stride=2, activation="elu", dilation_rate=1, depth_activation=False,
                               return_skip=True)

    # Level - 4 Xception down
    x, skip4 = _xception_block(x, [512, 512, 512], "L4", skip_conn_type="conv",
                               stride=2, activation="elu", dilation_rate=1, depth_activation=False,
                               return_skip=True)
    # Bottleneck
    x = _xception_block(x, [1024, 1024, 1024], "Bottleneck_entry", skip_conn_type="none",
                        stride=1, activation="elu", dilation_rate=1, depth_activation=False)
    x = _xception_block(x, [1024, 1024, 1024], "Bottleneck_Res0", skip_conn_type="sum",
                        stride=1, activation="elu", dilation_rate=1, depth_activation=False)
    x = _xception_block(x, [1024, 1024, 1024], "Bottleneck_Res1", skip_conn_type="sum",
                        stride=1, activation="elu", dilation_rate=1, depth_activation=False)
    x = _xception_block(x, [1024, 1024, 1024], "Bottleneck_Res2", skip_conn_type="sum",
                        stride=1, activation="elu", dilation_rate=1, depth_activation=False)
    x = tf.keras.layers.SpatialDropout2D(0.3)(x)

    # Level - 4 up
    x = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), activation="elu", name="U4_up")(x)
    U4_attn, U4_map = layers.IndividualAttentionLayer(512, name="U4_attn")(skip4, x)
    x = tf.keras.layers.Concatenate(axis=-1, name="U4_merge")([x, U4_attn])
    x = _xception_block(x, [512, 512, 512], "U4", skip_conn_type="conv",
                        stride=1, activation="elu", dilation_rate=1, depth_activation=True)
    # Level - 3 up
    x = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), activation="elu", name="U3_up")(x)
    U3_attn, U3_map = layers.IndividualAttentionLayer(256, name="U3_attn")(skip3, x)
    x = tf.keras.layers.Concatenate(axis=-1, name="U3_merge")([x, U3_attn])
    x = _xception_block(x, [256, 256, 256], "U3", skip_conn_type="conv",
                        stride=1, activation="elu", dilation_rate=1, depth_activation=True)

    # Level - 2 up
    x = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), activation="elu", name="U2_up")(x)
    U2_attn, U2_map = layers.IndividualAttentionLayer(128, name="U2_attn")(skip2, x)
    x = tf.keras.layers.Concatenate(axis=-1, name="U2_merge")([x, U2_attn])
    x = _xception_block(x, [128, 128, 128], "U2", skip_conn_type="conv",
                        stride=1, activation="elu", dilation_rate=1, depth_activation=True)

    # Level - 1 up
    x = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), activation="elu", name="U1_up")(x)
    U1_attn, U1_map = layers.IndividualAttentionLayer(64, name="U1_attn")(skip1, x)
    x = tf.keras.layers.Concatenate(axis=-1, name="U1_merge")([x, U1_attn])
    x = _conv_block(x, 64, 3, "elu", name="U1", batch_norm=True)

    y = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", padding="same", name="output_layer")(x)

    return tf.keras.Model(inputs=[inp], outputs=[y, U1_map, U2_map, U3_map, U4_map])
