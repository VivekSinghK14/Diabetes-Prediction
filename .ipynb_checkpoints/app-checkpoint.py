from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array([features])
    features = scaler.transform(features)
    
    prediction = model.predict(features)[0]
    result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)
