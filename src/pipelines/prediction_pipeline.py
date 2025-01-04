import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, load_h5_model
import sys

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Load preprocessor and model
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.h5')

            preprocessor = load_object(preprocessor_path)
            model = load_h5_model(model_path)

            # Apply preprocessing to the features
            data_scaled = preprocessor.transform(features)

            # Make prediction
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 mean_radius: float,
                 mean_texture: float,
                 mean_perimeter: float,
                 mean_area: float,
                 mean_smoothness: float,
                 mean_compactness: float,
                 mean_concavity: float,
                 mean_concave_points: float,
                 mean_symmetry: float,
                 mean_fractal_dimension: float,
                 radius_error: float,
                 texture_error: float,
                 perimeter_error: float,
                 area_error: float,
                 smoothness_error: float,
                 compactness_error: float,
                 concavity_error: float,
                 concave_points_error: float,
                 symmetry_error: float,
                 fractal_dimension_error: float,
                 worst_radius: float,
                 worst_texture: float,
                 worst_perimeter: float,
                 worst_area: float,
                 worst_smoothness: float,
                 worst_compactness: float,
                 worst_concavity: float,
                 worst_concave_points: float,
                 worst_symmetry: float,
                 worst_fractal_dimension: float):
        
        self.data = {
            'mean radius': mean_radius,
            'mean texture': mean_texture,
            'mean perimeter': mean_perimeter,
            'mean area': mean_area,
            'mean smoothness': mean_smoothness,
            'mean compactness': mean_compactness,
            'mean concavity': mean_concavity,
            'mean concave points': mean_concave_points,
            'mean symmetry': mean_symmetry,
            'mean fractal dimension': mean_fractal_dimension,
            'radius error': radius_error,
            'texture error': texture_error,
            'perimeter error': perimeter_error,
            'area error': area_error,
            'smoothness error': smoothness_error,
            'compactness error': compactness_error,
            'concavity error': concavity_error,
            'concave points error': concave_points_error,
            'symmetry error': symmetry_error,
            'fractal dimension error': fractal_dimension_error,
            'worst radius': worst_radius,
            'worst texture': worst_texture,
            'worst perimeter': worst_perimeter,
            'worst area': worst_area,
            'worst smoothness': worst_smoothness,
            'worst compactness': worst_compactness,
            'worst concavity': worst_concavity,
            'worst concave points': worst_concave_points,
            'worst symmetry': worst_symmetry,
            'worst fractal dimension': worst_fractal_dimension
        }

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = self.data
            df = pd.DataFrame([custom_data_input_dict])
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)