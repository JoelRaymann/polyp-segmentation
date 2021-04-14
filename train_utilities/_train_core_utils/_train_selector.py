"""
Script that defines a train selector which selects the train loop suitable for the given input model
name
"""
# Import necessary packages
import tensorflow as tf

# Import self packages
from ._unet_dice import train_unet_dice
from ._resunet import train_resunet
from ._fcn8 import train_fcn8
from ._deeplabv3 import train_deeplabv3
from ._unet_attn import train_unet_attn
from ._unet_guided_attn import train_unet_guided_attn
from ._gar_net import train_gar_net
from ._segnet import train_segnet
from ._dilated_resfcn import train_dilated_resfcn
from ._gar_net_exp import train_gar_net_exp


def train(model_name: str, model: tf.keras.Model, dataset, epoch_steps: int, metrics_tracker: tuple, optimizer=None,
          **kwargs):
    """
    Function to train a selected model with a train loop. This also supports a test loop too although
    please use it for validation alone.

    Parameters
    ----------
    model : str
        The name of the model used here.
    model_name : tf.keras.Model
        A built model for training.
    dataset : tf.data.Dataset
        A TF pipeline based dataset loop
    epoch_steps : int
        The maximum steps to take for the training loop to get terminated
    metrics_tracker : tuple
        The metrics as tuple to track and update. Expecting : (loss, f1_score, iou_score, dice_score)
    optimizer : tf.keras.optimizers.Optimizer
        The keras optimizer to use for training. NOTE: if this is set as None, then the loop is set to do a inference
        loop A.K.A test loop

    Returns
    -------
    tuple
        Results of all the metrics passed in the tracker (loss, f1_score, iou_score, dice_score)
    """

    # Check Models and train accordingly
    if model_name == "UNet-Dice":
        return train_unet_dice(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    elif model_name in ["ResUNet", "ResUNet++", "SE-UNet", "Dilated-UNet"]:
        return train_resunet(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    elif model_name == "FCN8":
        return train_fcn8(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    elif model_name == "SegNet":
        return train_segnet(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    elif model_name == "DeepLabv3":
        return train_deeplabv3(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    elif model_name == "UNet-Attn":
        return train_unet_attn(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    elif model_name == "UNet-GuidedAttn":
        return train_unet_guided_attn(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    elif model_name == "Dilated-ResFCN":
        return train_dilated_resfcn(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    elif model_name == "GAR-Net-Experimental":
        return train_gar_net_exp(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    elif model_name == "GAR-Net":
        return train_gar_net(model, dataset, epoch_steps, metrics_tracker, optimizer, **kwargs)

    else:
        print("[ERROR]: {0} model doesn't exist or is still not supported".format(model_name))
        raise NotImplementedError
