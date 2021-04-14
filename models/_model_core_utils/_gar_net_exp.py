"""
Experimental Works
"""
import tensorflow as tf
import layers

class ResidualIdentityBlock(tf.keras.layers.Layer):

    def __init__(self, kernel_size: int, filters: list, activation="relu", **kwargs):
        super(ResidualIdentityBlock, self).__init__(**kwargs)

        # Get the list of filters
        filters1, filters2, filters3 = filters

        # Define the blocks
        self.conv1 = layers.Conv2DBN(filters=filters1,
                                     kernel_size=1,
                                     activation=activation)
        self.conv2 = layers.Conv2DBN(filters=filters2,
                                     kernel_size=kernel_size,
                                     padding="same",
                                     activation=activation)
        self.conv3 = layers.Conv2DBN(filters=filters3,
                                     kernel_size=1,
                                     activation=None)
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        # Add the shortcut
        x = tf.keras.layers.add([x, inputs])
        return self.activation(x)


class ResidualIdentityDilationBlock(tf.keras.layers.Layer):

    def __init__(self,
                 kernel_size: int,
                 filters: list,
                 dilation_rate: int or list,
                 activation="relu",
                 **kwargs):
        super(ResidualIdentityDilationBlock, self).__init__(**kwargs)

        # Get the list of filters
        filters1, filters2, filters3 = filters

        # Define the blocks
        self.conv1 = layers.Conv2DBN(filters=filters1,
                                     kernel_size=1,
                                     activation=activation)
        self.conv2 = layers.Conv2DBN(filters=filters2,
                                     kernel_size=kernel_size,
                                     dilation_rate=dilation_rate,
                                     padding="same",
                                     activation=activation)
        self.conv3 = layers.Conv2DBN(filters=filters3,
                                     kernel_size=1,
                                     activation=None)
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        # Add the shortcut
        x = tf.keras.layers.add([x, inputs])
        return self.activation(x)


class ResidualStrideBlock(tf.keras.layers.Layer):

    def __init__(self,
                 kernel_size: int,
                 filters: list,
                 activation="relu",
                 strides=(2, 2),
                 **kwargs):
        super(ResidualStrideBlock, self).__init__(**kwargs)

        # Get the filters
        filters1, filters2, filters3 = filters

        # Define the operations
        self.conv1 = layers.Conv2DBN(filters=filters1,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=activation)
        self.conv2 = layers.Conv2DBN(filters=filters2,
                                     kernel_size=kernel_size,
                                     padding="same",
                                     activation=activation)
        self.conv3 = layers.Conv2DBN(filters=filters3,
                                     kernel_size=1,
                                     activation=None)

        self.shortcut = layers.Conv2DBN(filters=filters3,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=None)
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        # shortcut
        x_shortcut = self.shortcut(inputs)

        # add
        x = tf.keras.layers.add([x, x_shortcut])
        return self.activation(x)


class ResidualDilatedStrideBlock(tf.keras.layers.Layer):

    def __init__(self,
                 kernel_size: int,
                 filters: list,
                 dilation_rate: int or list,
                 activation="relu",
                 strides=(2, 2),
                 **kwargs):
        super(ResidualDilatedStrideBlock, self).__init__(**kwargs)

        # Get the filters
        filters1, filters2, filters3 = filters

        # Define the operations
        self.conv1 = layers.Conv2DBN(filters=filters1,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=activation)
        self.conv2 = layers.Conv2DBN(filters=filters2,
                                     kernel_size=kernel_size,
                                     dilation_rate=dilation_rate,
                                     padding="same",
                                     activation=activation)
        self.conv3 = layers.Conv2DBN(filters=filters3,
                                     kernel_size=1,
                                     activation=None)

        self.shortcut = layers.Conv2DBN(filters=filters3,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=None)
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        # shortcut
        x_shortcut = self.shortcut(inputs)

        # add
        x = tf.keras.layers.add([x, x_shortcut])
        return self.activation(x)

def GARNetExperimental(input_size=(256, 256, 3), test_mode=False):

    assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))

    image_width, image_height, n_channels = input_size

    inp = tf.keras.layers.Input(shape=(image_width, image_height, n_channels), dtype="float32", name="input_layer")

    # Starting Conv Block
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name="conv_start_pad")(inp)
    x = layers.Conv2DBN(64, 7, strides=(2, 2), padding="valid", name="conv_start", activation="relu")(x)

    # ENCODER BLOCK
    # Level 1 Down Block
    x = ResidualStrideBlock(3, [64, 64, 128], strides=(1, 1), activation="relu", name="l1_res_stride")(x)
    x = ResidualIdentityBlock(3, [64, 64, 128], activation="relu", name="l1_res_1")(x)
    x = ResidualIdentityBlock(3, [64, 64, 128], activation="relu", name="l1_res_2")(x)
    skip1 = layers.Conv2DBN(64, 1, padding="same", activation="relu", name="skip1_projection")(x)

    # Level 2 Down Block
    x = ResidualStrideBlock(3, [64, 64, 256], strides=(2, 2), activation="relu", name="l2_res_stride")(x)
    x = ResidualIdentityBlock(3, [64, 64, 256], activation="relu", name="l2_res_1")(x)
    x = ResidualIdentityBlock(3, [64, 64, 256], activation="relu", name="l2_res_2")(x)
    skip2 = layers.Conv2DBN(64, 1, padding="same", activation="relu", name="skip2_projection")(x)

    # Level 3 Down Block
    x = ResidualStrideBlock(3, [128, 128, 512], strides=(2, 2), activation="relu", name="l3_res_stride")(x)
    x = ResidualIdentityBlock(3, [128, 128, 512], activation="relu", name="l3_res_1")(x)
    x = ResidualIdentityBlock(3, [128, 128, 512], activation="relu", name="l3_res_2")(x)
    x = ResidualIdentityBlock(3, [128, 128, 512], activation="relu", name="l3_res_3")(x)
    skip3 = layers.Conv2DBN(128, 1, padding="same", activation="relu", name="skip3_projection")(x)

    # Level 4 Down Block
    x = ResidualStrideBlock(3, [256, 256, 1024], strides=(2, 2), activation="relu", name="l4_res_stride")(x)
    x = ResidualIdentityBlock(3, [256, 256, 1024], activation="relu", name="l4_res_1")(x)
    x = ResidualIdentityBlock(3, [256, 256, 1024], activation="relu", name="l4_res_2")(x)
    x = ResidualIdentityBlock(3, [256, 256, 1024], activation="relu", name="l4_res_3")(x)
    x = ResidualIdentityBlock(3, [256, 256, 1024], activation="relu", name="l4_res_4")(x)
    x = ResidualIdentityBlock(3, [256, 256, 1024], activation="relu", name="l4_res_5")(x)
    skip4 = layers.Conv2DBN(256, 1, padding="same", activation="relu", name="skip4_projection")(x)

    # BOTTLENECK
    x = ResidualDilatedStrideBlock(3, [512, 512, 2048],
                                   strides=(1, 1),
                                   dilation_rate=(2, 2),
                                   activation="relu", name="btn_res_dilated_stride")(x)
    x = ResidualIdentityDilationBlock(3, [512, 512, 2048], dilation_rate=(2, 2), activation="relu",
                                      name="btn_dilated_res_1")(x)
    x = ResidualIdentityDilationBlock(3, [512, 512, 2048], dilation_rate=(4, 4), activation="relu",
                                      name="btn_dilated_res_2")(x)
    x = layers.Conv2DBN(256, 1, padding="same", activation="relu", name="btn_output_projection")(x)
    btn_out = tf.keras.layers.UpSampling2D((16, 16), interpolation="bilinear", name="btn_out")(x)

    # DECODER BLOCK
    # Level 4 up
    u4_attn, u4_map = layers.AttentionLayer(256, name="u4_attn")([skip4, x])
    u4_out = tf.keras.layers.UpSampling2D((16, 16), interpolation="bilinear", name="u4_output")(u4_attn)

    # Level 3 up
    u4_attn = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), activation=None)(u4_attn)
    u3_attn, u3_map = layers.AttentionLayer(128, name="u3_attn")([skip3, u4_attn])
    u3_out = tf.keras.layers.UpSampling2D((8, 8), interpolation="bilinear", name="u3_output")(u3_attn)

    # Level 2 up
    u3_attn = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), activation=None)(u3_attn)
    u2_attn, u2_map = layers.AttentionLayer(64, name="u2_attn")([skip2, u3_attn])
    u2_out = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear", name="u2_output")(u2_attn)

    # Level 1 up
    u2_attn = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), activation=None)(u2_attn)
    u1_attn, u1_map = layers.AttentionLayer(64, name="u1_attn")([skip1, u2_attn])
    u1_out = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear", name="u1_output")(u1_attn)

    # Merge All
    out = tf.keras.layers.Concatenate(axis=-1, name="output_concat")([btn_out, u4_out, u3_out, u2_out, u1_out])
    out = layers.Conv2DBN(256, 3, padding="same", activation="relu", name="output_conv_1")(out)
    out = layers.Conv2DBN(128, 3, padding="same", activation="relu", name="output_conv_2")(out)
    out = layers.Conv2DBN(1, 1, padding="same", activation="sigmoid", name="output_layer")(out)

    return tf.keras.Model(inputs=[inp], outputs=[out, u1_map, u2_map, u3_map, u4_map])




# def GARNetExperimental(input_size=(256, 256, 3), test_mode=False):
#
#     assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))
#
#     image_width, image_height, n_channels = input_size
#
#     inp = tf.keras.layers.Input(shape=(image_width, image_height, n_channels), dtype="float32", name="input_layer")
#
#     # Level 1
#     x = layers.Conv2DBN(64, 3, activation="elu", padding="same", name="conv_entry_1")(inp)
#     x = layers.Conv2DBN(64, 3, activation="elu", padding="same", name="conv_entry_2")(x)
#     skip1 = layers.Conv2DBN(64, 3, activation="elu", padding="same", dilation_rate=(16, 16), name="rb1_skip")(x)
#
#     # Level 2
#     x = ResidualStrideBlock(3, [64, 64, 256], activation="elu", name="rb2_stride_1")(x)
#     x = ResidualIdentityBlock(3, [64, 64, 256], activation="elu", name="rb2_conv_1")(x)
#     x = ResidualIdentityBlock(3, [64, 64, 256], activation="elu", name="rb2_conv_2")(x)
#     skip2 = layers.Conv2DBN(64, 3, activation="elu", padding="same", dilation_rate=(8, 8), name="rb2_skip")(x)
#
#     # Level 3
#     x = ResidualStrideBlock(3, [128, 128, 512], activation="elu", name="rb3_stride_1")(x)
#     x = ResidualIdentityBlock(3, [128, 128, 512], activation="elu", name="rb3_conv_1")(x)
#     x = ResidualIdentityBlock(3, [128, 128, 512], activation="elu", name="rb3_conv_2")(x)
#     x = ResidualIdentityBlock(3, [128, 128, 512], activation="elu", name="rb3_conv_3")(x)
#     skip3 = layers.Conv2DBN(256, 3, activation="elu", padding="same", dilation_rate=(4, 4), name="rb3_skip")(x)
#
#     # Level 4
#     x = ResidualStrideBlock(3, [256, 256, 1024], activation="elu", name="rb4_stride_1")(x)
#     x = ResidualIdentityBlock(3, [256, 256, 1024], activation="elu", name="rb4_conv_1")(x)
#     x = ResidualIdentityBlock(3, [256, 256, 1024], activation="elu", name="rb4_conv_2")(x)
#     x = ResidualIdentityBlock(3, [256, 256, 1024], activation="elu", name="rb4_conv_3")(x)
#     x = ResidualIdentityBlock(3, [256, 256, 1024], activation="elu", name="rb4_conv_4")(x)
#     x = ResidualIdentityBlock(3, [256, 256, 1024], activation="elu", name="rb4_conv_5")(x)
#     skip4 = layers.Conv2DBN(256, 3, padding="same", activation="elu")(x)
#
#     # Bottle-neck
#     x = ResidualDilatedStrideBlock(3, [512, 512, 2048], dilation_rate=2, strides=(1, 1), activation="elu", name="rb_bn_stride_1")(x)
#     x = ResidualIdentityDilationBlock(3, [512, 512, 2048], dilation_rate=2, activation="elu", name="rb_bn_conv_2")(
#         x)
#     x = ResidualIdentityDilationBlock(3, [512, 512, 2048], dilation_rate=4, activation="elu", name="rb_bn_conv_3")(
#         x)
#     x = layers.Conv2DBN(256, 1, activation="elu")(x)
#
#     # UP Level 4
#     U4_attn, U4_map = layers.AttentionLayer(256, name="U4_attn")(skip4, x)
#     u4 = tf.keras.layers.UpSampling2D((8, 8), interpolation="bilinear")(U4_attn)
#
#     # UP Level 3
#     x = tf.keras.layers.Conv2DTranspose(256, 1, strides=(2, 2), activation=None)(x)
#     x = tf.keras.layers.BatchNormalization(axis=-1)(x)
#     x = tf.keras.layers.Activation("elu")(x)
#     U3_attn, U3_map = layers.AttentionLayer(256, name="U3_attn")(skip3, x)
#     u3 = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(U3_attn)
#
#     # UP Level 4
#     x = tf.keras.layers.Conv2DTranspose(64, 1, strides=(2, 2), activation=None)(x)
#     x = tf.keras.layers.BatchNormalization(axis=-1)(x)
#     x = tf.keras.layers.Activation("elu")(x)
#     U2_attn, U2_map = layers.AttentionLayer(64, name="U2_attn")(skip2, x)
#     u2 = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(U2_attn)
#
#     # UP Level 4
#     x = tf.keras.layers.Conv2DTranspose(64, 1, strides=(2, 2), activation=None)(x)
#     x = tf.keras.layers.BatchNormalization(axis=-1)(x)
#     x = tf.keras.layers.Activation("elu")(x)
#     U1_attn, U1_map = layers.AttentionLayer(64, name="U1_attn")(skip1, x)
#
#     x = tf.keras.layers.Concatenate(axis=-1)([u4, u3, u2, U1_attn, x])
#     x = layers.Conv2DBN(512, 1, activation="elu")(x)
#     x = layers.Conv2DBN(256, 1, activation="elu")(x)
#     out = layers.Conv2DBN(1, 1, activation="sigmoid", name="output_layer")(x)
#
#     return tf.keras.Model(inputs=[inp], outputs=[out, U1_map, U2_map, U3_map, U4_map])