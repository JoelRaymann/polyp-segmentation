"""
Script Implementing Dilated ResFCN data generator core functionality and all its helper functions
"""

# Import necessary packages
import tensorflow as tf
import numpy as np

# Import in-house packages
import weight_map_utils as wmu


# Handling Dilated ResFCN
def _load_data_dilated_resfcn(img, seg):
    """
    Function to prepare the images and return the data in accordance with the Dilated ResFCN model for training.

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

    # categorize seg
    seg = tf.keras.utils.to_categorical(seg, num_classes=2, dtype="float32")

    return img, seg
