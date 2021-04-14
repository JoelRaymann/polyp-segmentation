"""
Script to implement the Squeeze Excitation Block
"""
import tensorflow as tf


class SqueezeExcitationBlock(tf.keras.layers.Layer):
    """
    A class that implements the squeeze excitation block
    """

    def __init__(self, ratio: int, **kwargs):
        """
        A class that implements the squeeze excitation block.

        Parameters
        ----------
        ratio: int,
            The ratio param in the (n_channel / ratio) formula of SE Block.
        kwargs
        """
        super(SqueezeExcitationBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape: tuple):
        """
        Function to build the layer using the input_shape

        Parameters
        ----------
        input_shape: tuple
            the shape of the input tensor
        """
        # Get the channels from the input shape
        n_channels = input_shape[-1]

        # build the layer
        self.global_avg = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(int(n_channels // self.ratio), activation="relu")
        self.fc2 = tf.keras.layers.Dense(n_channels, activation="sigmoid")
        self.reshape = tf.keras.layers.Reshape((1, 1, n_channels))

    def call(self, x):
        """
        Function to run the forward pass for the layer
        """
        out = self.global_avg(x)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.reshape(out)
        return tf.keras.layers.multiply([x, out])

    def get_config(self):
        return {
            "ratio": self.ratio
        }
