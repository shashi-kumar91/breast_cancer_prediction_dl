import pickle
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.logger import logging
from tensorflow.keras.models import load_model

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object successfully saved at {file_path}")
    except Exception as e:
        logging.error(f"Failed to save object at {file_path}: {e}")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            logging.info(f"Object successfully loaded from {file_path}")
            return pickle.load(file_obj)
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        raise CustomException(f"File not found: {file_path}", sys)
    except Exception as e:
        logging.error(f"Exception occurred in load_object function: {e}")
        raise CustomException(e, sys)

def save_h5_model(file_path, model):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        model.save(file_path)
        logging.info(f"H5 model successfully saved at {file_path}")
    except Exception as e:
        logging.error(f"Failed to save H5 model at {file_path}: {e}")
        raise CustomException(e, sys)

def load_h5_model(file_path):
    try:
        model = load_model(file_path)
        logging.info(f"H5 model successfully loaded from {file_path}")
        return model
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        raise CustomException(f"File not found: {file_path}", sys)
    except Exception as e:
        logging.error(f"Exception occurred in load_h5_model function: {e}")
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training and evaluating model: {model_name}")
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics for train and test datasets
            metrics = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
                "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
                "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
                "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
            }

            report[model_name] = metrics
            logging.info(f"Metrics for {model_name}: {metrics}")

        return report

    except Exception as e:
        logging.error(f"Exception occurred during model evaluation: {e}")
        raise CustomException(e, sys)