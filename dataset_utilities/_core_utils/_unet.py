"""
Script Implementing U-Net's data generator core functionality and all its helper functions
"""

# Import necessary packages
import tensorflow as tf
import numpy as np

# Import in-house packages
import weight_map_utils as wmu


# Handling U-Net
def _load_data_unet(img, seg):
    """
    Function to prepare the images and return the data in accordance with the U-Net model for training.

    Parameters
    ----------
    img : tf.Tensor
        The image input tensor.
    seg : tf.Tensor
        The segmentation input tensor

    Returns
    -------
    tuple
        The tuple output (image, weight_map, segmentation)
    """
    # Get the numpy version
    img = img.numpy().copy()
    seg = seg.numpy().copy()

    # Normalize
    img = img / 255.0
    seg = seg / 255.0

    # categorize seg
    seg = tf.keras.utils.to_categorical(seg, num_classes=2, dtype="float32")

    # Find the wmap
    wmap = np.zeros_like(seg)
    for index in range(seg.shape[-1]):
        wmap[:, :, index] = wmu.get_weight_map(seg[:, :, index])

    return img, wmap, seg