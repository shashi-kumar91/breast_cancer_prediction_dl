from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
from src.logger import logging
import sys

application = Flask(__name__)
app = application

# Home page route
@app.route('/')
def home_page():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            # Create an instance of CustomData with user inputs from the form
            data = CustomData(
                mean_radius=float(request.form.get('mean radius', 0.0)),
                mean_texture=float(request.form.get('mean texture', 0.0)),
                mean_perimeter=float(request.form.get('mean perimeter', 0.0)),
                mean_area=float(request.form.get('mean area', 0.0)),
                mean_smoothness=float(request.form.get('mean smoothness', 0.0)),
                mean_compactness=float(request.form.get('mean compactness', 0.0)),
                mean_concavity=float(request.form.get('mean concavity', 0.0)),
                mean_concave_points=float(request.form.get('mean concave points', 0.0)),
                mean_symmetry=float(request.form.get('mean symmetry', 0.0)),
                mean_fractal_dimension=float(request.form.get('mean fractal dimension', 0.0)),
                radius_error=float(request.form.get('radius error', 0.0)),
                texture_error=float(request.form.get('texture error', 0.0)),
                perimeter_error=float(request.form.get('perimeter error', 0.0)),
                area_error=float(request.form.get('area error', 0.0)),
                smoothness_error=float(request.form.get('smoothness error', 0.0)),
                compactness_error=float(request.form.get('compactness error', 0.0)),
                concavity_error=float(request.form.get('concavity error', 0.0)),
                concave_points_error=float(request.form.get('concave points error', 0.0)),
                symmetry_error=float(request.form.get('symmetry error', 0.0)),
                fractal_dimension_error=float(request.form.get('fractal dimension error', 0.0)),
                worst_radius=float(request.form.get('worst radius', 0.0)),
                worst_texture=float(request.form.get('worst texture', 0.0)),
                worst_perimeter=float(request.form.get('worst perimeter', 0.0)),
                worst_area=float(request.form.get('worst area', 0.0)),
                worst_smoothness=float(request.form.get('worst smoothness', 0.0)),
                worst_compactness=float(request.form.get('worst compactness', 0.0)),
                worst_concavity=float(request.form.get('worst concavity', 0.0)),
                worst_concave_points=float(request.form.get('worst concave points', 0.0)),
                worst_symmetry=float(request.form.get('worst symmetry', 0.0)),
                worst_fractal_dimension=float(request.form.get('worst fractal dimension', 0.0)),
            )

            # Convert the data to DataFrame
            final_new_data = data.get_data_as_dataframe()

            # Initialize the prediction pipeline and make a prediction
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)

            # Round the prediction result
            results = round(pred[0], 2)

            # Return the result to the form
            return render_template('form.html', final_result=results)
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            return render_template('form.html', final_result="Error occurred during prediction.")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
