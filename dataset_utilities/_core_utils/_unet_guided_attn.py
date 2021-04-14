"""
Scripts to prepare the data for UNet Guided Attention Architecture
"""
# Import the necessary packages
import tensorflow as tf
from cv2 import cv2 as cv


# Handling U-Net Guided Attention
def _load_data_unet_guided_attn(img, seg):
    """
    Function to prepare the images and return the data in accordance with the U-Net Guided Attention dice model for
    training.

    Parameters
    ----------
    img : tf.Tensor
        The image input tensor.
    seg : tf.Tensor
        The segmentation input tensor

    Returns
    -------
    tuple
        A tuple of (img, seg, attn1_map, attn2_map, attn3_map, attn4_map)
    """
    # Get the numpy version
    img = img.numpy().copy()
    seg = seg.numpy().copy()

    # Image_size
    map_width, map_height = seg.shape

    # Resize the seg maps to get the attention maps
    attn1_map = seg.copy()
    attn2_map = cv.resize(seg, (map_width // 2, map_height // 2), interpolation=cv.INTER_CUBIC)
    attn3_map = cv.resize(seg, (map_width // 4, map_height // 4), interpolation=cv.INTER_CUBIC)
    attn4_map = cv.resize(seg, (map_width // 8, map_height // 8), interpolation=cv.INTER_CUBIC)

    # Level the maps
    attn1_map[attn1_map > 128] = 255
    attn1_map[attn1_map <= 128] = 0
    attn2_map[attn2_map > 128] = 255
    attn2_map[attn2_map <= 128] = 0
    attn3_map[attn3_map > 128] = 255
    attn3_map[attn3_map <= 128] = 0
    attn4_map[attn4_map > 128] = 255
    attn4_map[attn4_map <= 128] = 0

    # Normalize the maps
    img = img / 255.0
    seg = seg / 255.0
    attn1_map = attn1_map / 255.0
    attn2_map = attn2_map / 255.0
    attn3_map = attn3_map / 255.0
    attn4_map = attn4_map / 255.0

    return (img,
            tf.expand_dims(seg, axis=-1),
            tf.expand_dims(attn1_map, axis=-1),
            tf.expand_dims(attn2_map, axis=-1),
            tf.expand_dims(attn3_map, axis=-1),
            tf.expand_dims(attn4_map, axis=-1))
