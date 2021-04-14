"""
Script to include all metrics pertinent to the given segmentation task
"""
# import necessary packages
import tensorflow as tf
import tensorlayer as tl


def dice_coe(y_true, y_pred):
    """
    Function to find the dice coefficient of the given input and output

    Parameters
    ----------
    y_true : tf.Tensor
        The ground truth of the image. Supported dimension (None, W, H, 1)
    y_pred : tf.Tensor
        The predicted map of the image. Supported dimension (None, W, H, 1)

    Returns
    -------
    tf.Tensor
        The dice coefficient value for the given set
    """
    assert len(y_pred.shape) == 4 and len(y_true.shape) == 4, "[ERROR]: Input shape must be 4"

    return tl.cost.dice_coe(y_pred, y_true, loss_type="jaccard", axis=(1, 2, 3))


def iou_coe(y_true, y_pred):
    """
    Function to find the IoU coefficient of the given input and output

    Parameters
    ----------
    y_true : tf.Tensor
        The ground truth of the image. Supported dimension (None, W, H, 1)
    y_pred : tf.Tensor
        The predicted map of the image. Supported dimension (None, W, H, 1)

    Returns
    -------
    tf.Tensor
        The IoU coefficient value for the given set.
    """
    assert len(y_pred.shape) == 4 and len(y_true.shape) == 4, "[ERROR]: Input shape must be 4"

    return tl.cost.iou_coe(y_pred, y_true, axis=(1, 2, 3))
