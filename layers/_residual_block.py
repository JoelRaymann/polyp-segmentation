"""
Scripts to implement the Residual Block
"""
import tensorflow as tf

from ._conv_bn_layer import Conv2DBN


class ResidualBlock(tf.keras.layers.Layer):
    """
    A class that implements the Residual Block Layer
    """

    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 strides=(1, 1),
                 activation="relu",
                 dilation_rate=(1, 1),
                 activate_begin=False,
                 activate_end=False,
                 **kwargs):

        super(ResidualBlock, self).__init__(**kwargs)

        # set params
        self.activate_begin = activate_begin
        self.activate_end = activate_end

        if activate_begin:
            self.bn_start = tf.keras.layers.BatchNormalization(axis=-1)
            self.act_start = tf.keras.layers.Activation(activation)

        self.convbn = Conv2DBN(filters,
                               kernel_size,
                               strides=strides,
                               padding="same",
                               dilation_rate=dilation_rate,
                               activation=activation)

        self.conv = tf.keras.layers.Conv2D(filters,
                                           kernel_size,
                                           strides=strides,
                                           padding="same",
                                           dilation_rate=dilation_rate,
                                           activation="linear")

        if activate_end:
            self.bn_end = tf.keras.layers.BatchNormalization(axis=-1)
            self.act_end = tf.keras.layers.Activation(activation)

    def call(self, x, **kwargs):

        skip = x

        if self.activate_begin:
            x = self.bn_start(x)
            x = self.act_start(x)

        x = self.convbn(x)
        x = self.conv(x)

        if self.activate_end:
            x = self.bn_end(x)
            x = self.act_end(x)

        return tf.keras.layers.add([x, skip])
