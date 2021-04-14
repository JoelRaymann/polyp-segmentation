"""
Script to predict and save the output for each images in the test set of the dataset
"""

# Import necessary packages
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Import in-house packages
import models
import os_utilities
import dataset_utilities as du

# import within-house packages
from ._predict_core_utils import predict, plot


def predict_model(config: dict):
    """
    Function to predict the test dataset for the model using the given configuration from predict_config.yaml file.

    Parameters
    ----------
    config : dict
        The configuration read from predict_config.yaml file.

    Returns
    -------
    None
    """
    # Set GPU memory optimization
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        for index in range(len(physical_devices)):
            try:
                tf.config.experimental.set_memory_growth(physical_devices[index], True)

            except Exception as err:
                print("[WARN]: Failed to set memory growth for {0}".format(physical_devices[index]))
                print("[WARN]: Error", err, " .Skipping memory optimization")

    except Exception as err:
        print("[WARN]: memory optimization failed. Error:", err, " . Skipping!")

    # Set up random states
    np.random.seed(100)
    random.seed(100)
    tf.random.set_seed(100)

    # Get the required configurations
    test_batch_size = config["test_batch_size"]

    model_name = config["model_name"]

    # Create the dataset
    dataset_path = config["dataset_path"]
    dataset_family = config["dataset_family"]

    # get width and height
    image_width = config["image_width"]
    image_height = config["image_height"]

    # Handle attn outputs
    attn_output = bool(config["attn_output"])

    model_name = model_name + "_" + dataset_family

    print("[INFO]: Using Configuration: \n", config)

    # Load the dataset
    print("[INFO]: Loading dataset")
    if dataset_family == "CVC-ClinicDB":
        X, y = du.get_cvc_clinic_datapath(dataset_path)
    elif dataset_family == "Kvasir-Seg":
        X, y = du.get_kvasir_seg_datapath(dataset_path)
    else:
        print("[ERROR]: {0} dataset family is unrecognized or not supported!".format(dataset_family))
        raise NotImplementedError

    X_train, X_sided, y_train, y_sided = train_test_split(X, y, random_state=100, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_sided, y_sided, random_state=100, test_size=0.5)

    print("[INFO]: Testing set size: ", len(X_test))

    # Loading the model
    model_save_path = "./outputs/model_save/{0}/".format(model_name)
    print("[INFO]: Building the model - {0}".format(model_name))
    model = models.ModelSelector(config)
    print("[INFO]: Loading Best saved model:")
    model.load_weights(model_save_path + "best_model/best_{0}.h5".format(model_name))

    print("[INFO]: Loading the test set")
    test_datagen = du.DataGenerator(X_test,
                                    y_test,
                                    image_size=(image_width, image_height),
                                    model_name=config["model_name"],
                                    batch_size=test_batch_size,
                                    dataset_family=dataset_family,
                                    initial_size=None,
                                    aug_config_path=None,
                                    shuffle=False)

    print("[INFO]: Loading TF data")
    test_dataset = test_datagen.get_tf_data()
    test_steps = len(test_datagen)

    # Get input X and y
    x = np.copy(np.asarray(test_datagen._dataset_X))
    y_gnd = np.copy(np.asarray(test_datagen._dataset_y))
    y_gnd = y_gnd / 255.0

    # PREDICT
    print("[INFO]: Predicting")
    if attn_output:
        y_pred, attn1_pred, attn2_pred, attn3_pred, attn4_pred = predict(model,
                                                                         dataset=test_dataset,
                                                                         test_steps=test_steps,
                                                                         attn_outputs=attn_output)

        y_pred = np.copy(np.asarray(y_pred, dtype="float32"))
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = 0.0

        folders = [
            "./outputs/output_visualization/{0}/input/".format(model_name),
            "./outputs/output_visualization/{0}/ground/".format(model_name),
            "./outputs/output_visualization/{0}/prediction/".format(model_name),
            "./outputs/output_visualization/{0}/attn1/".format(model_name),
            "./outputs/output_visualization/{0}/attn2/".format(model_name),
            "./outputs/output_visualization/{0}/attn3/".format(model_name),
            "./outputs/output_visualization/{0}/attn4/".format(model_name),
        ]
        print("[INFO]: Setting up folders for dumping plots: ")
        os_utilities.make_directories(paths=folders)

        root_output_path = "./outputs/output_visualization/{0}/".format(model_name)

        # Plot and dump
        for index in tqdm(range(test_batch_size * test_steps)):

            plot(x[index],
                 output_path="{0}/input/test_{1}.png".format(root_output_path, index + 1),
                 cmap=None)
            plot(y_gnd[index],
                 output_path="{0}/ground/test_gnd_{1}.png".format(root_output_path, index + 1),
                 cmap="gray")
            plot(y_pred[index, :, :, 0],
                 output_path="{0}/prediction/test_pred_{1}.png".format(root_output_path, index + 1),
                 cmap="gray")
            plot(attn1_pred[index, :, :, 0],
                 output_path="{0}/attn1/test_attn1_{1}.png".format(root_output_path, index + 1),
                 cmap="jet")
            plot(attn2_pred[index, :, :, 0],
                 output_path="{0}/attn2/test_attn2_{1}.png".format(root_output_path, index + 1),
                 cmap="jet")
            plot(attn3_pred[index, :, :, 0],
                 output_path="{0}/attn3/test_attn3_{1}.png".format(root_output_path, index + 1),
                 cmap="jet")
            plot(attn4_pred[index, :, :, 0],
                 output_path="{0}/attn4/test_attn4_{1}.png".format(root_output_path, index + 1),
                 cmap="jet")

    else:
        y_pred = predict(model,
                         dataset=test_dataset,
                         test_steps=test_steps,
                         attn_outputs=attn_output)

        y_pred = np.copy(np.asarray(y_pred, dtype="float32"))
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = 0.0

        folders = [
            "./outputs/output_visualization/{0}/input/".format(model_name),
            "./outputs/output_visualization/{0}/ground/".format(model_name),
            "./outputs/output_visualization/{0}/prediction/".format(model_name),
        ]
        print("[INFO]: Setting up folders for dumping plots: ")
        os_utilities.make_directories(paths=folders)

        root_output_path = "./outputs/output_visualization/{0}/".format(model_name)

        # Plot and dump
        for index in tqdm(range(test_batch_size * test_steps)):

            plot(x[index],
                 output_path="{0}/input/test_{1}.png".format(root_output_path, index + 1),
                 cmap=None)
            plot(y_gnd[index],
                 output_path="{0}/ground/test_gnd_{1}.png".format(root_output_path, index + 1),
                 cmap="gray")
            plot(y_pred[index, :, :, 0],
                 output_path="{0}/prediction/test_pred_{1}.png".format(root_output_path, index + 1),
                 cmap="gray")

    print("[INFO]: Wrapping up :-)")
    print("[INFO]: Success")
    return None
