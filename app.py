import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def init_model():
    """Initialize or load the model"""
    global model, scaler
    try:
        # Try to load existing model
        model = joblib.load('sgd_model.joblib')
        scaler = joblib.load('scaler.joblib')
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading model: {e}. Creating new model...")
        # Create and train new model with sample data
        model = SGDClassifier(max_iter=1000, random_state=42)
        scaler = StandardScaler()
        
        # Sample training data
        X = np.random.rand(100, 4)  # 4 features
        y = np.random.randint(0, 2, 100)  # Binary classification
        
        # Fit scaler and model
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        # Save the model and scaler
        joblib.dump(model, 'sgd_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')

# Initialize model on startup
init_model()

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate form data
        required_fields = ['cgpa', 'internship_count', 'english_score', 'project_count']
        for field in required_fields:
            if field not in request.form:
                raise ValueError(f"Missing required field: {field}")

        # Convert form data to integers
        try:
            features = [float(request.form[field]) for field in required_fields]
        except ValueError:
            raise ValueError("All fields must be numeric values")

        # Validate ranges
        if not (0 <= features[0] <= 10):  # CGPA
            raise ValueError("CGPA must be between 0 and 10")
        if not (0 <= features[1] <= 10):  # Internship count
            raise ValueError("Internship count must be between 0 and 10")
        if not (0 <= features[2] <= 100):  # English score
            raise ValueError("English score must be between 0 and 100")
        if not (0 <= features[3] <= 20):  # Project count
            raise ValueError("Project count must be between 0 and 20")

        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'
        
        return render_template('index.html', 
                             prediction_text=f'Prediction: {output}',
                             success=True)

    except ValueError as e:
        return render_template('index.html', 
                             error=str(e),
                             success=False)
    except Exception as e:
        return render_template('error.html', 
                             error="An unexpected error occurred. Please try again.",
                             success=False)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == "__main__":
    app.run(debug=True)