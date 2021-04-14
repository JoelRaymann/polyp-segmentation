# Consist of utils to read/write files
# import necessary packages
import numpy as np
from skimage.io import imshow
from matplotlib import pyplot as plt
from PIL import Image

# NOTE: Read/Plot images
def read_binary_image(image_path: str):
    """
    Function to read the binary map given in the
    image path and return the np.array of the image

    Parameters
    ----------
    image_path {str} -- The path of the image to read

    Returns {np.array} -- The np.array of the image
    -------
    """
    img = Image.open(image_path, mode = "r")
    img = np.copy(np.asarray(img, dtype = "float32"))

    # Binarize the map
    img[img == 0] = 0
    img[img > 1] = 1

    return img

def save_plot(image, to_save: str, figsize = (10, 10), title = None):
    """
    Function to save the heatmap plot of the
    generated weight map.

    Parameters
    ----------
    image {np.array} -- The image to plot and save
    to_save {str} -- The path to save the plot
    figsize {tuple} -- The tuple to mention the size
    of the plot to save. default = (10, 10)
    title {str} : The title of the plot

    Returns None
    -------
    """
    # Set the plot
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, fontdict={"fontsize": figsize[0] + 5, "fontweight": "bold"})
    plt.imshow(image, cmap="jet")
    plt.colorbar()
    plt.savefig(to_save, bbox_inches = "tight")
    plt.close()

# write/load npz files
def save_npz(image, to_save: str):
    """
    Function to save the image
    in the given destination as a
    npz format for uncompressed
    recovery

    Parameters
    ----------
    image {np.array} -- The image to save
    to_save {str} -- The path to save the image
    as npz

    Returns None
    -------
    """
    np.savez(to_save, image)
    return None

def load_npz(path: str, return_index = -1):
    """
    Function to load the .npz file
    and return the numpy object

    Parameters
    ----------
    path {str}: Path to the npz file
    return_index {int}: Mention return_index
    to return a specific index. if value is -1
    then it will return the raw numpy data

    Returns {np.array}: Returns the numpy
    array.
    -------
    """
    data = np.load(path)
    if return_index == -1:
        return np.array(data.values())
    else:
        return data["arr_{0}".format(return_index)]