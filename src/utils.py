import os, sys, time
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException
import dill, json

def save_model(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            st = time.time()  # Start time for performance tracking
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            logging.info(f"Evaluation initiated for {model_name}.")

            para = param[model_name]
            gs = GridSearchCV(model, para, cv=10, verbose=1, n_jobs=-1)
            logging.info(f"GridSearchCV initiated for {model_name}.")
            gs.fit(x_train, y_train)
            logging.info(f"GridSearchCV fit completed for {model_name}.")

            model.set_params(**gs.best_params_)
            logging.info(f"Best params set for {model_name}: {gs.best_params_}")

            # Fit model with best parameters
            model.fit(x_train, y_train)
            logging.info(f"Fitting completed for {model_name}.")

            # Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Calculating accuracy
            logging.info(f"Calculating accuracy for {model_name}.")
            train_model_accuracy = accuracy_score(y_true=y_train, y_pred=y_train_pred)
            test_model_accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)

            report[model_name] = test_model_accuracy
            end = time.time()  # End time
            elapsed_time = end - st  # Calculate elapsed time
            logging.info(f"{model_name} completed in {elapsed_time:.2f} seconds with accuracy {test_model_accuracy}.")
        
        return report

    except Exception as e:
        raise CustomException(e, sys)

def save_json_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)
