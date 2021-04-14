"""
Implement weighted cross-entropy here
"""

# Import the necessary packages
import tensorflow as tf


# Implement the function
def weighted_crossentropy(y_true, y_pred, weight_map, **kwargs):
    """
    Function to implement the weighted cross-entropy based on the ideas of U-Net paper.

    Parameters
    ----------
    y_true : tf.Tensor
        The ground truth tensor map
    y_pred : tf.Tensor
        The predicted tensor map
    weight_map : tf.Tensor
        The tensor weight map
    kwargs

    Returns
    -------
    tf.Tensor
        The loss tensor for the given samples
    """
    # Get the epsilon
    _epsilon = tf.keras.backend.epsilon()

    # Normalize the softmax output
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True, name="normalize_output")

    # Clip activation with epsilon to avoid zeros -> this is needed to avoid log(0) situations.
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon, name="add_epsilon")

    # log activate the predictions
    y_pred = tf.math.log(y_pred, name="log_activation")

    # multiply the weight map
    y_pred_weighted = tf.math.multiply(y_pred, weight_map)

    # Aggregate the loss pixel-wise and return it
    return -tf.reduce_sum(y_true * y_pred_weighted, axis=-1, name="pixel_wise_sum")
