"""
Script to implement the convolution with batch normalization
"""
import tensorflow as tf


class Conv2DBN(tf.keras.layers.Layer):
    """
    A class that implements basic Conv2D with batch normalization
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 strides=(1, 1),
                 padding="valid",
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 batch_norm_axis=-1,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=0.001,
                 **kwargs):
        super(Conv2DBN, self).__init__(**kwargs)

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            activation="linear",
        )
        self.bn = tf.keras.layers.BatchNormalization(axis=batch_norm_axis,
                                                     momentum=batch_norm_momentum,
                                                     epsilon=batch_norm_epsilon)
        if activation is not None:
            self.activation = tf.keras.layers.Activation(activation)
        else:
            self.activation = None

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            return self.activation(x)
        else:
            return x