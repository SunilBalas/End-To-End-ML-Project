#Common functionalities
import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

def split_train_test_data(df:pd.DataFrame, test_size:float, random_state:int=None) -> tuple:
    '''
        This function takes a DataFrame, a test size ratio, and an optional random state
        and splits the data into training and testing sets.

        Parameters:
            df (pandas.DataFrame): The input DataFrame containing the dataset to be split
            test_size (float): The proportion of the dataset to include in the testing set
            (Should be a float value between 0.0 and 1.0)
            random_state (int, optional): An integer seed for the random number generator
            (If provided, it ensures reproducibility in the random split. Default is None)

        Returns:
            A tuple containing the following elements:
                train_set (pandas.DataFrame): The training data
                test_set (pandas.DataFrame): The testing data
    '''
    try:
        logging.info("Splitting the train and test data initiated")
        
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
        return (
            train_set,
            test_set
        )
    except Exception as ex:
        raise CustomException(ex, sys)
    
def save_object(file_path:str, obj:ColumnTransformer):
    '''
        This function takes a preprocessor object, typically used for data preprocessing in
        machine learning tasks, and saves it to a pickle file for later use.

        Parameters:
            file_path (str): The file path, including the .pkl extension, where the preprocessor
            object will be stored
            preprocessor: The data preprocessor object (e.g., scikit-learn transformer) to be saved
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)

        logging.info("Pickle file saved")
    except Exception as ex:
        raise CustomException(ex, sys)
    