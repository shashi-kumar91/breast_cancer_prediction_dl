# Basic Import
import numpy as np
import pandas as pd
import tensorflow as tf
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.utils import save_object, evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.h5')  # Using .h5 format for saving the model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting training and test data')
            X_train, X_valid, y_train, y_valid = train_test_split(train_arr[:, :-1], train_arr[:, -1], test_size=0.2, random_state=42)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info('Building the model')
            LAYERS = [
                tf.keras.layers.Flatten(input_shape=X_train.shape[1:], name="inputLayer"),
                tf.keras.layers.Dense(40, activation="relu", name="hiddenLayer1"),
                tf.keras.layers.Dense(40, activation="relu", name="hiddenLayer2"),
                tf.keras.layers.Dense(1, activation="sigmoid", name="outputLayer")
            ]
            
            model_clf = tf.keras.models.Sequential(LAYERS)
            
            # Compile the model
            model_clf.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

            # Train the model
            logging.info('Training the model with training data')
            history = model_clf.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

            # Evaluate the model on the test data
            logging.info('Evaluating the model on test data')
            test_loss, test_acc = model_clf.evaluate(X_test, y_test)
            logging.info(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')

            # Save the trained model
            save_object(self.model_trainer_config.trained_model_file_path, model_clf)
            logging.info(f'Model saved at {self.model_trainer_config.trained_model_file_path}')

        except Exception as e:
            logging.error('Error occurred in Model Training')
            raise CustomException(e, sys)