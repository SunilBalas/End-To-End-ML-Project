#Common functionalities
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import dill

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def split_train_test_data(df:pd.DataFrame, test_size:float, random_state:int=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    
def save_object(file_path:str, obj:ColumnTransformer) -> None:
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
    
def evaluate_models(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, models:Dict[str, object]) -> Dict[str, float]:
    '''
        Train Regression Models and Return R2 scores for test data.

        Parameters:
            - X_train (numpy.ndarray): Feature matrix for training data
            - y_train (numpy.ndarray): Target vector for training data
            - X_test (numpy.ndarray): Feature matrix for testing data
            - y_test (numpy.ndarray): Target vector for testing data
            - models (dict): A dictionary where keys are model names and values are instantiated regression models

        Returns:
            - report (dict): A dictionary where keys are model names and values are the R2 scores on the test data
    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)