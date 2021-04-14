"""
Scripts to implement the global attention upsample layer
"""
import tensorflow as tf


class GlobalAttentionUpsample(tf.keras.layers.Layer):
    """
    A class to implement the Global Attention Up-sampling
    """
    def __init__(self, upsample_rate=(2, 2), **kwargs):
        """
        A class to implement the Global Attention Up-sampling.

        Parameters
        ----------
        upsample_rate: tuple
            The upsampling rate for the upsample process
        kwargs
        """

        super(GlobalAttentionUpsample, self).__init__(**kwargs)
        self.upsample_rate=upsample_rate

    def build(self, input_shapes):
        low_shapes, high_shapes = input_shapes

        # get the low level and high level channels
        low_channels, high_channels = low_shapes[-1], high_shapes[-1]

        self.conv_low = tf.keras.Sequential([
            tf.keras.layers.Conv2D(high_channels, 3, padding="same", activation="linear"),
            tf.keras.layers.BatchNormalization(axis=-1),
        ])

        self.pooling_features = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Reshape((1, 1, high_channels)),
            tf.keras.layers.Conv2D(high_channels, 1, activation="linear"),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Activation("relu")
        ])

        self.upsample = tf.keras.layers.UpSampling2D(size=self.upsample_rate, interpolation="bilinear")

    def call(self, x, **kwargs):
        x_low, x_high = x
        x_low = self.conv_low(x_low)
        high_level_features = self.pooling_features(x_high)
        x_low = tf.keras.layers.multiply([x_low, high_level_features])
        x_high = self.upsample(x_high)

        return tf.keras.layers.add([x_low, x_high])

