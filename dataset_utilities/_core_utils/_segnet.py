"""
Script Implementing SegNet data generator core functionality and all its helper functions
"""

# Import necessary packages
import tensorflow as tf


# Handling SegNet
def _load_data_segnet(img, seg):
    """
    Function to prepare the images and return the data in accordance with the SegNet model for training.

    Parameters
    ----------
    img : tf.Tensor
        The image input tensor.
    seg : tf.Tensor
        The segmentation input tensor

    Returns
    -------
    tuple
        The tuple output (image, segmentation)
    """
    # Get the numpy version
    img = img.numpy().copy()
    seg = seg.numpy().copy()

    # Normalize
    img = img / 255.0
    seg = seg / 255.0

    return img, tf.expand_dims(seg, axis=-1)