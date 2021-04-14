# Implement image read and dump if needed

# Import the necessary functions
from skimage.io import imsave
import numpy as np

# Define image writer
def save_image(save_path: str, image: np.ndarray):
    """
    Function to save the image of dimensions (W, H)
    or (W, H, C) or (W, H, C, A)

    Parameters
    ----------
    save_path {str} -- The save path to save the image
    image {np.ndarray} -- The image as np.ndarray

    Returns {bool} -- returns the status
    -------
    """

    if image.dtype != "uint8":
        image = image / 255. if np.max(image) > 1.0 else image
    imsave(save_path, image)
    return True
