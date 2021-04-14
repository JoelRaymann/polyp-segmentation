"""
Script to take a model and do the prediction for the given input
"""

# Import necessary packages
import tensorflow as tf
from tqdm import tqdm


def predict(model: tf.keras.Model, dataset: tf.data.Dataset, test_steps: int, attn_outputs: bool):
    """
    Function to predict and return the output for the given input image set X

    Parameters
    ----------
    model : tf.keras.Model
        The built model for the prediction.
    dataset : tf.keras.Dataset
        The setup X, y dataset for the prediction.
    test_steps : int
        The test steps a.k.a no. of batches for the testing set
    attn_outputs : bool, optional
        A status flag to indicate if the model produces attention outputs

    Returns
    -------
    list
        A list of predictions done for each batch
    """

    # Set the counter for batches
    batch_count = 0

    # Collect the predictions
    y_pred = list()
    if attn_outputs:
        attn1_pred = list()
        attn2_pred = list()
        attn3_pred = list()
        attn4_pred = list()

    # Loop and predict!
    try:
        with tqdm(total=test_steps) as pbar:
            for X, _ in dataset:
                # predict the output
                predictions = model(X, training=False)

                if attn_outputs:
                    pred, a1, a2, a3, a4 = predictions
                    attn1_pred.append(a1)
                    attn2_pred.append(a2)
                    attn3_pred.append(a3)
                    attn4_pred.append(a4)
                else:
                    pred = predictions

                y_pred.append(pred)

                batch_count += 1
                pbar.update(1)

                if batch_count >= test_steps:
                    y_pred = tf.keras.layers.concatenate(y_pred, axis=0)
                    if attn_outputs:
                        attn1_pred = tf.keras.layers.concatenate(attn1_pred, axis=0)
                        attn2_pred = tf.keras.layers.concatenate(attn2_pred, axis=0)
                        attn3_pred = tf.keras.layers.concatenate(attn3_pred, axis=0)
                        attn4_pred = tf.keras.layers.concatenate(attn4_pred, axis=0)
                        return y_pred, attn1_pred, attn2_pred, attn3_pred, attn4_pred
                    else:
                        return y_pred

    except Exception as err:
        print("[ERROR]: Error: ", err)
        exit(1)
