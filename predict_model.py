# import necessary packages
import argparse
import yaml

import predict_utilities

DESCRIPTION = """
This is a predict API
"""
VERSION = "0.01alpha"

if __name__ == "__main__":

    # For help
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    # Add options
    parser.add_argument("-V", "--version", help="Shows program version", action="store_true")
    parser.add_argument("-C", "--config-file", help="The YAML config file for prediction")
    # Read args
    args = parser.parse_args()

    # check for version
    if args.version:
        print("Using Version %s" % (VERSION))
        exit(1)

    if args.config_file:
        config_file_path = str(args.config_file)

    print("[INFO]: Parsing prediction Config file")
    config = yaml.load(open(config_file_path, mode="r"), Loader=yaml.FullLoader)

    print("[INFO]: The following configuration is parsed")
    for k, v in config.items():
        print("{0}: {1}".format(k, v))

    choice = int(input("[PROBE]: Do you wish to proceed (1/0): "))
    print(choice)
    if choice != 1:
        print("[INFO]: Aborting prediction operations")
        exit(1)

    else:
        print("[INFO]: Proceeding to predict using {0} model".format(config["model_name"]))
        predict_utilities.predict_model(config)
