# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model_path = 'sgd_model2.joblib'  # Updated file extension
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = 'Placed' if prediction[0] == 1 else 'Not Placed'

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
