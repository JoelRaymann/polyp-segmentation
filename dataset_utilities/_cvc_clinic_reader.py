"""
Script to help in reading the CVC Clinic Dataset for polyp
"""
# import necessary function
import os
import tifffile as tiff
from cv2 import cv2 as cv


# Functional APIs
def get_cvc_clinic_datapath(dataset_path: str) -> tuple:
    """
    Function to generate the X,y path list for the CVC Clinic dataset.

    Parameters
    ----------
    dataset_path : str
        The root path of the CVC-Clinic Dataset

    Returns
    -------
    tuple
        A tuple of list (X, y) where X is the list of original images and
         y is the list of ground truth images
    """
    # Get X
    X = ["{0}/Original/{1}".format(dataset_path, path) for path in os.listdir("{0}/Original/".format(dataset_path))]

    # Make y
    y = list(map(lambda x: x.replace("Original", "Ground Truth"), X))

    return X, y


def read_cvc_clinic_data(image_path: str, seg_path: str, resize=None) -> tuple:
    """
    Function to read the image and its segmentation map from the given
    image_path and seg_path.

    Parameters
    ----------
    image_path : str
        The path of the image to read.
    seg_path : str
        The path of the segmentation map to reads
    resize : tuple, optional
        Resize the images to the given tuple of (width, height) before returning

    Returns
    -------
    tuple
        A tuple of numpy arrays of (image, seg)
    """
    # Read the tiff images
    image = tiff.imread(image_path)
    seg = tiff.imread(seg_path)

    # Resize the images
    if resize is not None:

        assert type(resize) == tuple, "[ERROR]: Please set resize as a tuple (width, height)"
        image = cv.resize(image, (resize[0], resize[1]), interpolation=cv.INTER_CUBIC)
        seg = cv.resize(seg, (resize[0], resize[1]), interpolation=cv.INTER_CUBIC)

    # Handle value noise from map
    seg[seg >= 128] = 255
    seg[seg < 128] = 0

    return image, seg
