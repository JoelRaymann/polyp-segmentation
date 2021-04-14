"""
Scripts to plot the images
"""

# Import necessary packages
from matplotlib import pyplot as plt
import numpy as np


# Plot
def plot(image: np.ndarray, output_path: str, cmap: str):
    """
    Function to plot and save the image.

    Parameters
    ----------
    image : np.ndarray
        The image to plot and save.
    output_path : str
        The output path to save the model
    cmap : str
        The color map to use {None, "jet", "gray"}

    Returns
    -------
    None
    """

    # Plot the images
    plt.figure(num=1, figsize=(10, 10))
    plt.imshow(image, cmap=cmap)
    plt.grid(False)

    if cmap == "jet" or cmap == "viridis":
        plt.colorbar(shrink=0.8)
    else:
        plt.axis(False)

    plt.savefig(output_path, bbox_inches="tight")

    plt.clf()
    plt.close()
    return None
