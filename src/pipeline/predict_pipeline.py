import sys
from typing import Dict
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features:pd.DataFrame) -> any:
        '''
            Predicts target values using a trained machine learning model.

            Args:
                - features (pandas.DataFrame): The input features as a Pandas DataFrame

            Returns:
                - any: The predicted target values
        '''
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as ex:
            raise CustomException(ex, sys)
    
class CustomData:
    def __init__(self, request_data:Dict[str, any]) -> None:        
        self.gender = request_data["gender"]
        self.race_ethnicity = request_data["race_ethnicity"]
        self.parental_level_of_education = request_data["parental_level_of_education"]
        self.lunch = request_data["lunch"]
        self.test_preparation_course = request_data["test_preparation_course"]
        self.reading_score = request_data["reading_score"]
        self.writing_score = request_data["writing_score"]
        
    def get_data_as_data_frame(self) -> pd.DataFrame:
        '''
            Make the Data Frame from the requested data.
            
            Returns:
                - df(pandas.DataFrame): Returns the Data Frame of the data
        '''
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            
            return df
        except Exception as ex:
            raise CustomException(ex, sys)