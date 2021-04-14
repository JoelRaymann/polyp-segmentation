"""
IMPLEMENT A CUSTOM CSV LOGGER
"""

# Import necessary packages
import pandas as pd
import os


class CSVLogging:
    """
    A Class to build a callback for logging the given dictionary data in csv files
    NOTE: The dictionary data's key will be used as column_name and values will be
    used as column values. Values must be a list
    """
    def __init__(self, file_path: str, overwrite=False):
        """
        A Class to build a callback for logging the given dictionary data in csv files
        NOTE: The dictionary data's key will be used as column_name and values will be
        used as column values. Values must be a list

        Parameters
        ----------
        file_path : str
            The path to log the csv file.
        overwrite : bool
            A status flag to overwrite if there is an already existing file in the file_path
        """
        # Check for .csv extension
        if ".csv" not in file_path:
            file_path += ".csv"

        # Assign attributes
        self.file_path = file_path
        self.overwrite = overwrite

        # Check for overwriting
        if os.path.exists(file_path):
            print("[WARN]: Log already exist!!")
            if overwrite is True:
                print("[WARN]: Overwriting enabled! This data will be deleted!")
                os.remove(file_path)

    def log(self, data:dict) -> bool:
        """
        Logs the dictionary data as a csv file.NOTE: The dictionary data's key will be used as column_name
        and values will be used as column values. Values must be a list

        Parameters
        ----------
        data : dict
            The dictionary data as a path

        Returns
        -------
        bool
            The status flag for the operation
        """
        df = pd.DataFrame(data)
        if os.path.exists(self.file_path):
            df.to_csv(self.file_path, index=False, header=False, mode="a")
        else:
            df.to_csv(self.file_path, index=False, mode="w")
            self.overwrite = False
        return True
