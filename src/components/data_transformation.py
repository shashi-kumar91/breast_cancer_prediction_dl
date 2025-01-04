from sklearn.impute import SimpleImputer  # Handling Missing Values
from sklearn.preprocessing import StandardScaler  # Handling Feature Scaling
from sklearn.preprocessing import OneHotEncoder  # One-Hot Encoding for categorical features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')

            target_column_name = 'target'
            numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_columns.remove(target_column_name)

            categorical_columns = train_df.select_dtypes(include=[object]).columns.tolist()

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessing_obj = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing datasets')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error('Error occurred in Data Transformation')
            raise CustomException(e, sys)