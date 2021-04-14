"""
Scripts to include the segnet layers
"""
# import necessary packages
import tensorflow as tf


# Implement the MaxPooling with Indicing
class MaxPoolingWithIndicing2D(tf.keras.layers.Layer):
    """
    A Class that implements the keras layer for Max-pooling with indicing for the SegNet
    """
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="valid", **kwargs):
        """
        A Class that implements the keras layer for Max-pooling with indicing for the SegNet

        Parameters
        ----------
        pool_size: tuple, optional
            The pool size for the max-pooling operation (default is (2, 2)).
        strides: tuple, optional
            The striding for the max-pooling operation (default is (2, 2)).
        padding: str, optional
            The padding for the max-pooling operation ("valid" or "same") (default is "valid").
        kwargs
        """
        super(MaxPoolingWithIndicing2D, self).__init__(**kwargs)
        self.pooling_size = pool_size
        self.padding = padding.upper()
        self.strides = strides

    def call(self, inputs, **kwargs):

        output, argmax = tf.nn.max_pool_with_argmax(inputs,
                                                    ksize=[1, self.pooling_size[0], self.pooling_size[1], 1],
                                                    strides=[1, self.strides[0], self.strides[1], 1],
                                                    padding=self.padding)
        argmax = tf.cast(argmax, dtype="float32")
        return output, argmax

    def compute_output_shape(self, input_shape):

        if self.padding == "same":
            return input_shape, input_shape
        elif self.padding == "valid":
            output_shape = input_shape
            output_shape[1] = ((output_shape[1] - self.pooling_size[0]) // self.strides[0]) + 1
            output_shape[2] = ((output_shape[2] - self.pooling_size[1]) // self.strides[1]) + 1

            return [output_shape, output_shape]

        else:
            print("[ERROR]: Supported padding values 'same' or 'valid', got {0}".format(self.padding))
            raise NotImplementedError

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


# Implement the MaxUnpooling2D with Indicing
class MaxUnpooling2D(tf.keras.layers.Layer):
    """
    A Class that implements the keras layer for "Unpooling Max-pooling with indicing" for the SegNet
    """
    def __init__(self, size=(2, 2), **kwargs):
        """
        A Class that implements the keras layer for "Unpooling Max-pooling with indicingg for the SegNet

        Parameters
        ----------
        size: tuple, optional
            The kernel size for the max-unpooling operation (default is (2, 2)).
        kwargs

        NOTE: Beware no one can out pizza the hut!!
        """
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None, **kwargs):

        updates, mask = inputs[0], inputs[1]

        mask = tf.cast(mask, dtype="int32")
        input_shape = tf.shape(updates, out_type="int32")
        i_shape = updates.shape
        o_shape = (input_shape[0], i_shape[1] * self.size[0], i_shape[2] * self.size[1], i_shape[3])

        # Calculate the new shape
        if output_shape is None:
            output_shape = (input_shape[0],
                            input_shape[1] * self.size[0],
                            input_shape[2] * self.size[1],
                            input_shape[3])
        self.output_shape1 = output_shape

        # calculate indices for batch, height, width and feature_maps
        one_like_mask = tf.ones_like(mask, dtype="int32")
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(
            tf.range(output_shape[0], dtype='int32'),
            shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(
            tf.stack([b, y, x, f]),
            [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, o_shape)
        return ret

    def compute_output_shape(self, input_shape):

        mask_shape = input_shape[1]
        return (mask_shape[0],
                mask_shape[1]*self.size[0],
                mask_shape[2]*self.size[1],
                mask_shape[3])
