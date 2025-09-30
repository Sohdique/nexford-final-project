# app.py - Flask application for customer segmentation predictions using the trained KMeans model

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

SCALER_PATH = 'notebooks/output/scaler.pkl'
MODEL_PATH = 'notebooks/output/kmeans_model.pkl'
COLUMNS_PATH = 'notebooks/output/feature_columns.pkl'

if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Scaler or model file not found. Ensure the notebook has been run to generate them.")

scaler = joblib.load(SCALER_PATH)
kmeans = joblib.load(MODEL_PATH)

if os.path.exists(COLUMNS_PATH):
    TRAINED_COLUMNS = joblib.load(COLUMNS_PATH)
else:
    raise FileNotFoundError("Feature columns file not found. Run training notebook first.")

CATEGORICAL_COLUMNS = [
    'ProductCategory', 'ProductBrand', 'ProductPrice', 'CustomerAge',
    'CustomerGender', 'PurchaseFrequency', 'CustomerSatisfaction', 'PurchaseIntent']

EXPECTED_COLUMNS = [
    'ProductCategory', 'ProductBrand', 'ProductPrice', 'CustomerAge',
    'CustomerGender', 'PurchaseFrequency', 'CustomerSatisfaction', 'PurchaseIntent'
]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            cleaned_data[key] = value[0] if value else None
        else:
            cleaned_data[key] = value
    
    try:
        input_df = pd.DataFrame([cleaned_data]) 
    except ValueError as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in input_df.columns]
    if missing_cols:
        return jsonify({'error': f'Missing columns: {", ".join(missing_cols)}'}), 400

    encoded_df = pd.get_dummies(input_df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    
    # if 'ProductID' in encoded_df.columns:
    #     encoded_df = encoded_df.drop(columns=['ProductID'])
    
    for col in TRAINED_COLUMNS:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    
    encoded_df = encoded_df[TRAINED_COLUMNS]

    if encoded_df.empty or len(encoded_df.columns) == 0:
        return jsonify({'error': 'No valid features after encoding'}), 400
    
    scaled_data = scaler.transform(encoded_df)
    
    cluster_label = kmeans.predict(scaled_data)[0]
    return jsonify({'cluster': int(cluster_label)})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Customer Segmentation Prediction API. Use /predict endpoint with POST.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)