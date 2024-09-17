import os,sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
from dataclasses import dataclass
from src.utils import save_model, evaluate_models, save_json_object
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig():
    model_path = os.path.join('artifacts','model.pkl')
    model_report_path = os.path.join('artifacts','models_report.json')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            x_train,y_train,x_test,y_test = (train_array[:,:-1],train_array[:,-1],
                                             test_array[:,:-1],test_array[:,-1])

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "AdaboostClassifier":AdaBoostClassifier(),
                "Gradient Boosting Classifier":GradientBoostingClassifier(verbose=1),
                "Random Forest Classifier":RandomForestClassifier(verbose=1),
                "Support Vector Machine":SVC(verbose=True),
                "K Nearest Neighbours":KNeighborsClassifier(),
                "Naive Bayes":GaussianNB(),
                "Catboost Classifier":CatBoostClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "XGBoost Classifier":XGBClassifier()
            }
            params = {
                'Logistic Regression':{
                    'penalty':['elasticnet','l1','l2']
                },
                'Decision Tree':{
                    'max_depth':[10,20,30],
                    'min_samples_split':[2,5,10]
                },
                'AdaboostClassifier':{
                    'n_estimators':[100,150,200],
                    'learning_rate':[0.1,0.01,0.001]
                },
                'Gradient Boosting Classifier':{
                    'n_estimators':[100,150,200],
                    'max_depth':[10,20,30],
                    'learning_rate':[0.1,0.01,0.001]
                },
                'Random Forest Classifier':{
                    'n_estimators':[450],
                    'max_features':['log2'],
                    'max_depth':[340],
                    'min_samples_split':[3],
                    'min_samples_leaf':[8,10,12],
                    'criterion':['gini']
                },
                'Support Vector Machine':{
                    'kernel':['linear','poly','sigmoid','rbf'],
                    'gamma':['scale','auto']
                },
                'K Nearest Neighbours':{
                    'metric':['euclidean']
                },
                'Naive Bayes':{

                },
                'Catboost Classifier':{
                    'learning_rate':[0.1,0.01,0.001],
                    'depth':[10,20,30],
                    'iterations':[100,150,200],
                    'l2_leaf_reg':[2,3,4]
                },
                "XGBoost Classifier":{}
            }

            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            

            save_model(file_path=self.model_trainer_config.model_path,obj=best_model)

            predicted = best_model.predict(x_test)
            acc = accuracy_score(y_test,predicted)
            logging.info(f"best model : {best_model_name} on both training and testing data with accuracy {acc}")
            save_json_object(file_path=self.model_trainer_config.model_report_path,obj=model_report)
            return(acc,best_model_name)
        except Exception as e:
            raise CustomException(e,sys)