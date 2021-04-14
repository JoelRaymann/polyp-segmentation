"""
Scripts to train a single loop of ResUNet
"""

# Import necessary packages
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm

# Import in-house packages
import metrics


def _run_resunet(model, x, y_gnd, training=True):
    # Gradient Tape it
    with tf.GradientTape() as tape:
        # Run the model
        y_pred = model(x, training=training)

        # Find the loss
        loss_value = 1.0 - tl.cost.dice_coe(y_pred, y_gnd, loss_type="jaccard")
        loss_value += sum(model.losses)

    # Run the metric
    f1_score = tf.reduce_mean(tf.keras.metrics.binary_accuracy(y_gnd, y_pred))
    iou_coe = metrics.iou_coe(y_gnd, y_pred)
    dice_coe = metrics.dice_coe(y_gnd, y_pred)

    # Get gradients
    grad = tape.gradient(loss_value, model.trainable_variables)

    return loss_value, grad, (f1_score, iou_coe, dice_coe)


def train_resunet(model: tf.keras.Model, dataset: tf.data.Dataset, epoch_steps: int, metrics_tracker: tuple,
                    optimizer=None, **kwargs):
    """
    Function to train or test in a single loop in the ResUNet model for dice loss. NOTE: If optimizer is set as None,
    then the loop acts as a testing loop.

    Parameters
    ----------
    model : tf.keras.Model
        A built ResU-Net model for training.
    dataset : tf.data.Dataset
        A TF pipeline based dataset loop
    epoch_steps : int
        The maximum steps to take for the training loop to get terminated
    metrics_tracker : tuple
        The metrics as tuple to track and update
    optimizer : tf.keras.optimizers.Optimizer
        The keras optimizer to use for training. NOTE: if this is set as None, then the loop is set to do a inference
        loop A.K.A test loop

    Returns
    -------
    tuple
        returns the metrics value for displaying
    """

    # Set the counter for batches
    batch_count = 0

    # Track the metrics values
    loss_avg, f1_score_metric, dice_coe_metric, iou_coe_metric = metrics_tracker

    # Run the data loop
    try:
        with tqdm(total=epoch_steps) as pbar:
            for X, y in dataset:
                # optimize the model
                if optimizer is not None:
                    # Get the loss and predictions
                    loss_values, grad, metrics_eval = _run_resunet(model, X, y, training=True)
                    optimizer.apply_gradients(zip(grad, model.trainable_variables))

                else:
                    # Get the loss and predictions
                    loss_values, grad, metrics_eval = _run_resunet(model, X, y, training=False)

                # Get the metrics
                f1_score, iou_coe, dice_coe = metrics_eval

                # Culminate all the evaluation values
                loss_avg(loss_values)
                f1_score_metric(f1_score)
                dice_coe_metric(dice_coe)
                iou_coe_metric(iou_coe)

                # increase the batch count
                batch_count += 1
                pbar.set_postfix(loss=loss_avg.result().numpy(),
                                 f1_score=f1_score_metric.result().numpy(),
                                 iou_coe=iou_coe_metric.result().numpy(),
                                 dice_coe=dice_coe_metric.result().numpy())
                pbar.update(1)

                if batch_count >= epoch_steps:
                    loss_val, f1, iou, dice = (loss_avg.result(), f1_score_metric.result(), iou_coe_metric.result(),
                                               dice_coe_metric.result())

                    # Reset the metrics
                    loss_avg.reset_states()
                    f1_score_metric.reset_states()
                    iou_coe_metric.reset_states()
                    dice_coe_metric.reset_states()

                    return loss_val.numpy(), f1.numpy(), iou.numpy(), dice.numpy()

    except Exception as err:
        print("[ERROR]: Error: ", err)
        exit(1)
