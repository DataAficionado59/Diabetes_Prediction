import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig():
    raw_data_path:str = os.path.join('artifacts','raw_data.csv')
    train_data_path:str = os.path.join('artifacts','train_data.csv')
    test_data_path:str = os.path.join('artifacts','test_data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv('notebook\diabetes.csv')
            logging.info("Read the data")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Splitting of datta into train set and test set completed.")
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data complete.")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    ingestion_obj = DataIngestion()
    trainset,testset = ingestion_obj.initiate_data_ingestion()
    transformation_obj = DataTransformation()
    train_arr,test_arr = transformation_obj.initiate_data_transformation(trainset,testset)
    model_trainer_object = ModelTrainer()
    model_trainer_object.initiate_model_trainer(train_arr,test_arr)