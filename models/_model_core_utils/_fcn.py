"""
Script to implement the FCN8 Architecture
"""
import tensorflow as tf


def FCN8(input_size: tuple, test_mode=False):
    """
    Function to implement the FCN-8 Architecture for training and
    testing the model

    Parameters
    ----------
    config {dict} -- The configuration file for the training
    test_mode {bool} -- The status flag to compile the model for testing or training

    Returns {tf.keras.Model} -- A tensorflow's keras model of the FCN-8 compiled and ready for training.
    -------
    """
    # Get configuration
    assert len(input_size) == 3, "[ERROR]: Expected tuple of length 3 got {0}".format(len(input_size))

    image_width, image_height, n_channels = input_size

    # Define the model
    X = tf.keras.layers.Input(shape=(image_width, image_height, 3), dtype="float32", name="input_layer")

    # Down-sampling path
    # Level 1
    convL1_1 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="convL1_1")(X)
    convL1_2 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="convL1_2")(convL1_1)
    poolL1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="maxpoolL1")(convL1_2)

    # Level 2
    convL2_1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="convL2_1")(poolL1)
    convL2_2 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="convL2_2")(convL2_1)
    poolL2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="maxpoolL2")(convL2_2)

    # Level 3
    convL3_1 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="convL3_1")(poolL2)
    convL3_2 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="convL3_2")(convL3_1)
    convL3_3 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="convL3_3")(convL3_2)
    poolL3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="maxpoolL3")(convL3_3)

    # Level 4
    convL4_1 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="convL4_1")(poolL3)
    convL4_2 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="convL4_2")(convL4_1)
    convL4_3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="convL4_3")(convL4_2)
    poolL4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="maxpoolL4")(convL4_3)

    # Level 5
    convL5_1 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="convL5_1")(poolL4)
    convL5_2 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="convL5_2")(convL5_1)
    convL5_3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="convL5_3")(convL5_2)
    poolL5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="maxpoolL5")(convL5_3)

    # Level 6
    convL6 = tf.keras.layers.Conv2D(4096, 7, padding="same", activation="relu", name="fc6")(poolL5)

    # Level 7
    convL7 = tf.keras.layers.Conv2D(4096, 1, padding="same", activation="relu", name="fc7")(convL6)

    # Score
    score_fr = tf.keras.layers.Conv2D(2, 1, padding="same", activation="relu", name="score_fr")(convL7)

    # Transpose conv
    upU1 = tf.keras.layers.Conv2DTranspose(2, (2, 2), (2, 2), padding="valid", activation=None, name="score2")(score_fr)

    # Skip conv for pool4
    conv_poolL4 = tf.keras.layers.Conv2D(2, 1, padding="same", activation=None, name="score_poolL4")(poolL4)
    aggregate_score1 = tf.keras.layers.Add()([conv_poolL4, upU1])

    upU2 = tf.keras.layers.Conv2DTranspose(2, (2, 2), (2, 2), padding="valid", activation=None, name="score4")(aggregate_score1)

    # Skip conv for pool3
    conv_poolL3 = tf.keras.layers.Conv2D(2, 1, padding="same", activation=None, name="score_poolL3")(poolL3)
    aggregate_score2 = tf.keras.layers.Add()([conv_poolL3, upU2])

    # Up sampling
    upU3 = tf.keras.layers.Conv2DTranspose(1, (8, 8), (8, 8), padding="valid", activation="sigmoid", name="upsample")(aggregate_score2)

    return tf.keras.Model(inputs=X, outputs=upU3)
