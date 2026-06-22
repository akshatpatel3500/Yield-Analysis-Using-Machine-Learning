from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained RandomForest model, LabelEncoder, and Scaler
model = joblib.load('random_forest_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming the scaler is saved as 'scaler.pkl'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve values from form
    input_features = [float(x) for x in request.form.values()]

    # Reshape and Normalize the input features
    features = np.array([input_features])
    normalized_features = scaler.transform(features)  # Normalize the features

    # Model prediction on normalized features
    numeric_prediction = model.predict(normalized_features)

    # Convert numeric prediction to crop name using LabelEncoder
    crop_name = label_encoder.inverse_transform(numeric_prediction)[0]

    return render_template('index.html', prediction_text='Recommended Crop: {}'.format(crop_name))

if __name__ == "__main__":
    app.run(debug=True)
