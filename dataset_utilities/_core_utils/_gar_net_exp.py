"""
Script to prepare the data for the GAR-Net Experimental model
"""
import tensorflow as tf
from cv2 import cv2 as cv


# Handling GAR-Net
def _load_data_gar_net_exp(img, seg):
    """
    Function to prepare the images and return the data in accordance with the GAR-Net model for training.

    Parameters
    ----------
    img : tf.Tensor
        The image input tensor.
    seg : tf.Tensor
        The segmentation input tensor

    Returns
    -------
    tuple
        The tuple output (image, segmentation, attention_map1, attention_map2, attention_map3, attention_map4)
    """
    # Get the numpy version
    img = img.numpy().copy()
    seg = seg.numpy().copy()

    # preparing attention maps scale down
    image_width, image_height = seg.shape
    attn1 = cv.resize(seg, (image_width // 2, image_height // 2), interpolation=cv.INTER_CUBIC)
    attn2 = cv.resize(seg, (image_width // 4, image_height // 4), interpolation=cv.INTER_CUBIC)
    attn3 = cv.resize(seg, (image_width // 8, image_height // 8), interpolation=cv.INTER_CUBIC)
    attn4 = cv.resize(seg, (image_width // 16, image_height // 16), interpolation=cv.INTER_CUBIC)

    # Level the maps
    attn1[attn1 > 128] = 255
    attn1[attn1 <= 128] = 0
    attn2[attn2 > 128] = 255
    attn2[attn2 <= 128] = 0
    attn3[attn3 > 128] = 255
    attn3[attn3 <= 128] = 0
    attn4[attn4 > 128] = 255
    attn4[attn4 <= 128] = 0

    # Normalize
    img = img / 255.0
    seg = seg / 255.0

    attn1 = attn1 / 255.0
    attn2 = attn2 / 255.0
    attn3 = attn3 / 255.0
    attn4 = attn4 / 255.0

    return (img, tf.expand_dims(seg, axis=-1), tf.expand_dims(attn1, axis=-1), tf.expand_dims(attn2, axis=-1),
            tf.expand_dims(attn3, axis=-1), tf.expand_dims(attn4, axis=-1))
