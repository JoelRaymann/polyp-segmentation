"""
Script to implement the Atrous Spatial Pyramid Pooling
"""
# import necessary packages
import tensorflow as tf


class ASPP(tf.keras.Model):
    """
    A class the implements the ASPP layer called as
    Atrous Spatial Pyramid Pooling. Includes Image
    Pooling with it
    """

    def __init__(self, filters: int, atrous_rates: list, input_size: tuple, scale_rate: int, activation="elu",
                 epsilon=1e-5, **kwargs):
        """
        A class the implements the ASPP layer with batch_normalization layer support.

        Parameters
        ----------
        filters: int
            The number of filters to use for each spatial convolution.
        atrous_rates: int
            The list of atrous rates for each spatial convolution in the spatial pyramid.
        input_size: tuple
            The input width and height  of the original input image. This is necessary for the image pooling.
        scale_rate: int
            The number of depth reduced from the original input size. This can be calculated by i/p
            image_size/output_size.
        activation: str, optional
            The activation to use for each convolution layers (default: "elu")
        epsilon: float, optional
            The epsilon for the batchnormalization layer. (default is 1e-5)
        kwargs
        """
        super(ASPP, self).__init__(**kwargs)

        # 1x1 convolution
        self.conv_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, activation="linear"),
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=epsilon),
            tf.keras.layers.Activation(activation=activation)
        ])

        # Image Pooling
        self.img_pool = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(pool_size=(input_size[0] // scale_rate, input_size[1] // scale_rate)),
            tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, activation="linear"),
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=epsilon),
            tf.keras.layers.Activation(activation=activation),
            tf.keras.layers.Conv2DTranspose(filters, (input_size[0] // scale_rate, input_size[1] // scale_rate),
                                            strides=(input_size[0] // scale_rate, input_size[1] // scale_rate),
                                            activation="linear"),
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=epsilon),
            tf.keras.layers.Activation(activation=activation)
        ])

        # Apply Set of Atrous Convolutions
        self.atrous_conv_list = [tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, dilation_rate=atrous_rates[i], padding="same", activation="linear",
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=epsilon),
            tf.keras.layers.Activation(activation=activation)
        ]) for i in range(len(atrous_rates))]

    def call(self, x, **kwargs):

        # apply img_pooling and 1x1 conv
        aspp_outputs = [self.img_pool(x), self.conv_1(x)]

        # apply set of atrous convs
        for atrous_conv in self.atrous_conv_list:
            aspp_outputs.append(atrous_conv(x))

        # Concatenate and return
        return tf.keras.layers.Concatenate(axis=-1)(aspp_outputs)
