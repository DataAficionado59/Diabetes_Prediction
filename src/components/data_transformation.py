import os,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.utils import save_model

@dataclass
class DataTransformationConfig():
    preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # def get_data_transformer(self):
    #     try:
    #         numerical_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
            
    #         num_pipeline = Pipeline(
    #             steps=('scaler',StandardScaler())
    #         )
    #         preprocessor = ColumnTransformer(('num_pipeline',num_pipeline,numerical_columns))
    #         return preprocessor
            
    #     except Exception as e:
    #         raise CustomException(e,sys)

    def initiate_data_transformation(self,trainset_path,testset_path):
        try:
            train_df = pd.read_csv(trainset_path)
            test_df = pd.read_csv(testset_path)

            logging.info("Reading the train and test data completed now obtaining the preprocessor object")
            preprocessing_obj = StandardScaler()
            target_column_name = "Outcome"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df['Outcome']

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on both training and testing object.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info("Data Transformation complete now saving the preprocessing object")
            save_model(self.data_transformation_config.preprocessor_file_path,preprocessing_obj)
            return(
                train_arr,test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)