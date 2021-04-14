# Import the necessary packages
import numpy as np
from skimage import measure
from cv2 import cv2 as cv

def get_class_map(image):
    """
    Function to get the class weight map from
    the input image which is a numpy array of
    dim = (image_width, image_height)

    Parameters
    ----------
    image {np.array} -- The input image from which
    we will compute the class weight map

    Returns {np.array} -- The class weight map
    based on the concepts introduced in U-Net paper
    -------
    """
    class_weights = np.zeros(2, dtype="float32")

    # Count the pixels and calc the fraction of them
    class_weights[0] = 1.0 / ((image == 0).sum() + 1E-16) # Background
    class_weights[1] = 1.0 / ((image == 1).sum() + 1E-16) # Foreground
    class_weights /= class_weights.max() # Normalize the values

    # Generate the class_weight map
    cw_map = np.where(image == 0, class_weights[0], class_weights[1])
    return cw_map

def get_distance_map(image, w0 = 10.0, sigma = 5.0):
    """
    Function to get the distance weight map
    from the given images using the L2 distance.


    Parameters
    ----------
    image {np.array} -- The numpy array of the image
    to work with.
    w0 {int} -- The w0 value (refer formula)
    sigma {int} -- The sigma value (refer formula)

    Returns {np.array} -- The distance weight map to
    output
    -------
    """

    # Get total number of cells using component analysis
    cells = measure.label(image, connectivity = 2)

    # get the distance weight map
    dw_map = np.zeros_like(image, dtype = "float32")

    # Get individual cell maps
    maps = np.zeros((image.shape[0], image.shape[1], cells.max()))

    # Check for more than 2 component
    if cells.max() > 2:
        # for each cell
        for i in range(1, cells.max() + 1):
            # Get the distance matrix of the background pixels with the cell for all cells
            maps[:, :, i - 1] = cv.distanceTransform(1 - (cells == i).astype(np.uint8), cv.DIST_L2, 3)
        # Sort the maps according to the axis 2 to get the min distance in all the pixels
        maps = np.sort(maps, axis=2)
        # the nearest cell boundaries
        d1 = maps[:, :, 0]
        # the 2nd nearest cell boundaries
        d2 = maps[:, :, 1]
        # find the distance weight
        dis = ((d1 + d2) ** 2) / (2 * sigma * sigma)
        dw_map = w0 * np.exp(-dis) * (cells == 0)

    return dw_map

def get_weight_map(image, w0 = 10.0, sigma = 5.0):
    """
    Function to get the weight map for the given
    input image

    Parameters
    ----------
    image {np.array} -- The numpy array of the image
    to work with.
    w0 {int} -- The w0 value (refer formula)
    sigma {int} -- The sigma value (refer formula)

    Returns {np.array} -- The weight map to
    output
    -------
    """
    return get_class_map(image) + get_distance_map(image, w0, sigma)

