"""
Script to custom train the chosen model using Tensorflow custom train loop
"""

# Import necessary packages
import tensorflow as tf
import sys, traceback
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Import in-house packages
import models
import os_utilities
import dataset_utilities as du
import callbacks

# Import self scripts
from ._train_core_utils import train


def train_model(config: dict, load_weights=None, resume_epoch=None):
    """
    Function to train the model using the given configuration from train_config.yaml file.

    Parameters
    ----------
    config : dict
        The configuration read from train_config.yaml file
    load_weights : str, optional
        The path to load the weights. If None, then a fresh training is started. (default is None)
    resume_epoch : int, optional
        The epoch to resume training from if needed. If None, then a fresh training is started. (default is None)

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
    no_of_epochs = config["no_of_epochs"]
    steps_per_epoch = config["steps_per_epoch"] if config["steps_per_epoch"] > 0 else None

    train_batch_size = config["train_batch_size"]
    val_batch_size = config["val_batch_size"]
    test_batch_size = config["test_batch_size"]

    model_name = config["model_name"]

    # Create the dataset
    dataset_path = config["dataset_path"]
    dataset_family = config["dataset_family"]

    # get width and height
    image_width = config["image_width"]
    image_height = config["image_height"]
    initial_width = config["initial_width"]
    initial_height = config["initial_height"]

    model_name = model_name + "_" + dataset_family

    print("[INFO]: Using Configuration: \n", config)

    # Set up environments
    folders = [
        "./outputs/model_save/{0}/checkpoints/".format(model_name),
        "./outputs/output_logs/{0}/csv_log/".format(model_name),
        "./outputs/model_save/{0}/best_model/".format(model_name),
        "./outputs/model_save/{0}/saved_model/".format(model_name),
        "./outputs/output_logs/{0}/graphs/".format(model_name)
    ]
    print("[INFO]: Setting up folders: ")
    os_utilities.make_directories(paths=folders)

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

    print("[INFO]: Training set size: ", len(X_train))
    print("[INFO]: Validation set size: ", len(X_val))
    print("[INFO]: Testing set size: ", len(X_test))

    print("[INFO]: Loading Training set")
    train_datagen = du.DataGenerator(X_train,
                                     y_train,
                                     image_size=(image_width, image_height),
                                     model_name=config["model_name"],
                                     batch_size=train_batch_size,
                                     dataset_family=dataset_family,
                                     initial_size=(initial_width, initial_height),
                                     aug_config_path="./augmentation_config.yaml",
                                     shuffle=True)

    print("[INFO]: Loading Validation set")
    val_datagen = du.DataGenerator(X_val,
                                   y_val,
                                   image_size=(image_width, image_height),
                                   model_name=config["model_name"],
                                   batch_size=val_batch_size,
                                   dataset_family=dataset_family,
                                   initial_size=None,
                                   aug_config_path=None,
                                   shuffle=False)

    print("[INFO]: Setting tf.data pipeline")
    train_steps = len(train_datagen) if steps_per_epoch is None else int(steps_per_epoch)
    val_steps = len(val_datagen)

    train_dataset = train_datagen.get_tf_data()
    val_dataset = val_datagen.get_tf_data()

    # Get the model, loss and metrics
    print("[INFO]: Building the model - {0}".format(model_name))
    model = models.ModelSelector(config)

    # Load the weights if available
    if load_weights is not None:
        print("[INFO]: Load the weights from {0}".format(load_weights))
        model.load_weights(load_weights)

    # Setup Callbacks
    print("[INFO]: Setting up training Callbacks and Optimizers. Its almost done")
    resume_epoch = 0 if resume_epoch is None else resume_epoch
    overwrite = True if resume_epoch == 0 else False

    train_csv_logger = callbacks.CSVLogging("./outputs/output_logs/{0}/csv_log/train_log.csv".format(model_name),
                                            overwrite=overwrite)
    valid_csv_logger = callbacks.CSVLogging("./outputs/output_logs/{0}/csv_log/valid_log.csv".format(model_name),
                                            overwrite=overwrite)
    lr_reducer = callbacks.ReduceLROnPlateau(learning_rate=float(config["learning_rate"]),
                                             patience=4,
                                             decay_rate=1E-1,
                                             delta=0.0001,
                                             min_lr=1E-7,
                                             mode="min")

    model_save_path = "./outputs/model_save/{0}/".format(model_name)

    # Setup Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(config["learning_rate"]))

    # Check for monitor
    monitor_variable = 100.0  # Initialize a start max

    print("[INFO]: Setting up metrics")
    loss_avg = tf.keras.metrics.Mean(name="loss")
    f1_score_metric = tf.keras.metrics.Mean(name="f1_score")
    iou_coe_metric = tf.keras.metrics.Mean(name="iou_coe")
    dice_coe_metric = tf.keras.metrics.Mean(name="dice_coe")

    # Set up a custom train loop
    print("[INFO]: Beginning training loops")
    # Iterate epoch wise
    for epoch in range(resume_epoch, no_of_epochs):

        print("Training {0}/{1}".format(epoch + 1, no_of_epochs))

        try:
            # Training-loop == using batches
            train_loss, train_f1_score, train_iou, train_dice = train(config["model_name"],
                                                                      model,
                                                                      train_dataset,
                                                                      train_steps,
                                                                      metrics_tracker=(loss_avg,
                                                                                       f1_score_metric,
                                                                                       iou_coe_metric,
                                                                                       dice_coe_metric),
                                                                      optimizer=optimizer,
                                                                      epoch=epoch + 1,
                                                                      total_epochs=no_of_epochs)

            train_tracker = {
                "train_loss": [train_loss],
                "train_f1_score": [train_f1_score],
                "train_iou_coe": [train_iou],
                "train_dice_coe": [train_dice]
            }

            # Validation loop == using batches
            val_loss, val_f1_score, val_iou, val_dice = train(config["model_name"],
                                                              model,
                                                              val_dataset,
                                                              val_steps,
                                                              metrics_tracker=(loss_avg,
                                                                               f1_score_metric,
                                                                               iou_coe_metric,
                                                                               dice_coe_metric),
                                                              optimizer=None,
                                                              epoch=1,
                                                              total_epochs=1)
            val_tracker = {
                "val_loss": [val_loss],
                "val_f1_score": [val_f1_score],
                "val_iou_coe": [val_iou],
                "val_dice_coe": [val_dice]
            }

            model.save_weights(model_save_path + "checkpoints/{0}_ckpt.h5".format(model_name))
            print("[INFO]: Epoch {0}/{1} - \nTrain evaluation: {2}, \nValidation evaluation: {3}".
                  format(epoch + 1, no_of_epochs, train_tracker, val_tracker))
            train_csv_logger.log(train_tracker)
            valid_csv_logger.log(val_tracker)

            # Save the best model
            if monitor_variable > val_loss:
                monitor_variable = val_loss
                model.save_weights(model_save_path + "best_model/best_{0}.h5".format(model_name))

            # LR Reduce
            lr_reducer.check_lr(monitor_variable=val_loss, optimizer=optimizer)

        except KeyboardInterrupt:
            print("[INFO]: Interrupted Training. Trying to save model")
            model.save_weights(model_save_path + "{0}_{1}_interrupted.h5".format(model_name, epoch + 1))
            print("[INFO]: Attempting to run test on the best model so far!")
            break

        except Exception as err:
            print("[ERROR]: Unexpected Critical Error: ", err)
            print("[ERROR]: Trying to save the weights")
            model.save_weights(model_save_path + "{0}_{1}_critical.h5".format(model_name, epoch + 1))
            traceback.print_exc()
            sys.exit(2)

    print("[INFO]: Training completed. Saving model")
    model.save_weights(model_save_path + "saved_model/{0}.h5".format(model_name))

    print("[INFO]: Testing model")
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

    print("[INFO]: Testing Initiated")
    test_loss, test_f1_score, test_iou, test_dice = train(config["model_name"],
                                                          model,
                                                          test_dataset,
                                                          test_steps,
                                                          metrics_tracker=(loss_avg,
                                                                           f1_score_metric,
                                                                           iou_coe_metric,
                                                                           dice_coe_metric),
                                                          optimizer=None,
                                                          epoch=1,
                                                          total_epochs=1)
    test_tracker = {
        "test_loss": [test_loss],
        "test_f1_score": [test_f1_score],
        "test_iou_coe": [test_iou],
        "test_dice_coe": [test_dice]
    }
    print("[INFO]: Test Results: \n", test_tracker)
    with open("./outputs/output_logs/{0}/test_results.txt".format(model_name), mode="w") as f:
        f.write("Dumped Test Results for the model {0}\n".format(model_name))
        for k, v in test_tracker.items():
            f.write("{0} => {1}\n".format(k, v))

    print("[INFO]: Closing operations")
