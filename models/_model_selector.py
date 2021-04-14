"""
Script to develop a model selector for selection of models
"""

# Import necessary packages
import tensorflow as tf

# Import in-house models
from ._model_core_utils import UNet, UNetDice, ResUNet, FCN8, Deeplabv3, UNetAttn, GARNet, SegNet, ResUNetPlusPlus
from ._model_core_utils import DilatedResFCN, SEUNet, DilatedUNet, GARNetExperimental


def ModelSelector(config: dict, test_mode=False) -> tf.keras.Model:
    """
    Function to build the selected model for training and testing

    Parameters
    ----------
    config : dict
        The configuration dictionary read from train_config.yaml
    test_mode : bool
        The status flag to request building the model for deployment.

    Returns
    -------
    tf.keras.Model
        The keras model of the chosen architecture
    """
    # Get details
    model_name = config["model_name"]
    image_width = config["image_width"]
    image_height = config["image_height"]

    print("[INFO]: Selecting model: {0} with input size: {1}".format(model_name, (image_width, image_height, 3)))

    if model_name == "UNet-Dice":
        return UNetDice((image_width, image_height, 3), test_mode)

    elif model_name == "UNet":
        return UNet((image_width, image_height, 3), test_mode)

    elif model_name == "ResUNet":
        return ResUNet((image_width, image_height, 3), test_mode)

    elif model_name == "ResUNet++":
        return ResUNetPlusPlus((image_width, image_height, 3), test_mode)

    elif model_name == "DeepLabv3":
        return Deeplabv3(input_shape=(image_width, image_height, 3), classes=1, backbone="xception")

    elif model_name == "FCN8":
        return FCN8((image_width, image_height, 3), test_mode)

    elif model_name == "SegNet":
        return SegNet((image_width, image_height, 3), test_mode)

    elif model_name == "UNet-Attn" or model_name == "UNet-GuidedAttn":
        return UNetAttn((image_width, image_height, 3), test_mode)

    elif model_name == "Dilated-ResFCN":
        return DilatedResFCN((image_width, image_height, 3), test_mode)

    elif model_name == "SE-UNet":
        return SEUNet((image_width, image_height, 3), test_mode)

    elif model_name == "Dilated-UNet":
        return DilatedUNet((image_width, image_height, 3), test_mode)

    elif model_name == "GAR-Net-Experimental":
        return GARNetExperimental((image_width, image_height, 3), test_mode)

    elif model_name == "GAR-Net":
        return GARNet((image_width, image_height, 3), test_mode)

    else:
        raise NotImplementedError
