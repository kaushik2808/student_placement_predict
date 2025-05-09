import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def create_synthetic_data():
    """Generate synthetic data for training if no real data is available"""
    try:
        np.random.seed(42)
        n_samples = 1000
        data = {
            'cgpa': np.random.uniform(5.0, 10.0, n_samples),
            'internship_count': np.random.randint(0, 5, n_samples),
            'research_papers': np.random.randint(0, 4, n_samples),
            'project_count': np.random.randint(0, 6, n_samples),
            'programming_certifications': np.random.randint(0, 5, n_samples),
            'aptitude_score': np.random.uniform(50, 100, n_samples),
            'english_test_score': np.random.uniform(50, 100, n_samples)
        }
        df = pd.DataFrame(data)
        df['placed'] = ((df['cgpa'] > 7.5) & (df['internship_count'] > 1) & 
                       (df['aptitude_score'] > 70) & (df['english_test_score'] > 65)).astype(int)
        return df
    except Exception as e:
        raise Exception(f"Error generating synthetic data: {str(e)}")

def create_model():
    """Create and train a new model"""
    try:
        df = create_synthetic_data()
        X = df[['cgpa', 'internship_count', 'research_papers', 'project_count',
                'programming_certifications', 'aptitude_score', 'english_test_score']]
        y = df['placed']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler to 'model' directory in current working directory
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        return model, scaler
    except Exception as e:
        raise Exception(f"Error creating model: {str(e)}")

def load_model():
    """Load model and scaler from disk, or create new ones if they don't exist"""
    try:
        model_dir = 'model'
        model_path = os.path.join(model_dir, 'model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            model, scaler = create_model()
        
        return model, scaler
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def init_app():
    """Initialize the application"""
    global model, scaler
    try:
        model, scaler = load_model()
        print("Model and scaler initialized successfully")
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        raise e

# Initialize app on startup
init_app()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate form inputs
        required_fields = ['cgpa', 'internship_count', 'research_papers', 'project_count',
                         'programming_certifications', 'aptitude_score', 'english_test_score']
        data = {}
        for field in required_fields:
            if field not in request.form:
                return render_template('index.html', error=f"Missing field: {field}"), 400
            
            value = request.form[field]
            try:
                value = float(value)
            except ValueError:
                return render_template('index.html', error=f"Invalid value for {field}: must be a number"), 400
            
            # Validate ranges
            if field == 'cgpa' and not (0 <= value <= 10):
                return render_template('index.html', error="CGPA must be between 0 and 10"), 400
            elif field in ['internship_count', 'research_papers', 'project_count', 'programming_certifications'] and not (0 <= value <= 10):
                return render_template('index.html', error=f"{field} must be between 0 and 10"), 400
            elif field in ['aptitude_score', 'english_test_score'] and not (0 <= value <= 100):
                return render_template('index.html', error=f"{field} must be between 0 and 100"), 400
            
            data[field] = value
        
        # Prepare input for model
        input_data = np.array([[data['cgpa'], data['internship_count'], data['research_papers'],
                              data['project_count'], data['programming_certifications'],
                              data['aptitude_score'], data['english_test_score']]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Generate recommendations
        recommendations = []
        if data['cgpa'] < 7.5:
            recommendations.append("Improve your CGPA to above 7.5 for better placement chances.")
        if data['internship_count'] < 2:
            recommendations.append("Try to complete at least 2 internships.")
        if data['english_test_score'] < 70:
            recommendations.append("Work on your English proficiency to score above 70.")
        
        return render_template('result.html',
                             prediction='Placed' if prediction == 1 else 'Not Placed',
                             probability=round(probability * 100, 2),
                             recommendations=recommendations)
    except Exception as e:
        return render_template('error.html', error=f"Prediction error: {str(e)}"), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        required_fields = ['cgpa', 'internship_count', 'research_papers', 'project_count',
                         'programming_certifications', 'aptitude_score', 'english_test_score']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f"Missing field: {field}"}), 400
            
            try:
                value = float(data[field])
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': f"Invalid value for {field}: must be a number"}), 400
            
            # Validate ranges
            if field == 'cgpa' and not (0 <= value <= 10):
                return jsonify({'success': False, 'error': "CGPA must be between 0 and 10"}), 400
            elif field in ['internship_count', 'research_papers', 'project_count', 'programming_certifications'] and not (0 <= value <= 10):
                return jsonify({'success': False, 'error': f"{field} must be between 0 and 10"}), 400
            elif field in ['aptitude_score', 'english_test_score'] and not (0 <= value <= 100):
                return jsonify({'success': False, 'error': f"{field} must be between 0 and 100"}), 400
        
        # Prepare input for model
        input_data = np.array([[data['cgpa'], data['internship_count'], data['research_papers'],
                              data['project_count'], data['programming_certifications'],
                              data['aptitude_score'], data['english_test_score']]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'prediction_text': 'Placed' if prediction == 1 else 'Not Placed',
            'probability': round(probability, 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f"Prediction error: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    if model is None or scaler is None:
        return jsonify({'status': 'error', 'message': 'Model not initialized'}), 500
    return jsonify({'status': 'ok'})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True)
