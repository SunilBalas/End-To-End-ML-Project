# for training model
import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr:np.ndarray, test_arr:np.ndarray) -> float:
        '''
            Evaluates Regression Models and returns R2 Score for the best-fit model.

            Parameters:
                - train_arr (numpy.ndarray): A array containing the training data
                - test_arr (numpy.ndarray): A array containing the testing data

            Returns:
                - best_r2_score (float): The R2 Score of the best-fit model
        '''
        try:
            logging.info("Split train and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBoost Classifier": XGBRegressor(),
                "CatBoost Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }
            
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # fetch the model which has maximum accuracy
            best_model_score = max(sorted(model_report.values())) 
            
            # Get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6 :
                raise CustomException("No Best Model Found !")
            
            logging.info("Best found model on both train and test dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)
            
            return score
        except Exception as ex:
            raise CustomException(ex, sys)