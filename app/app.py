from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from joblib import load
import os

app = Flask(__name__)

MODEL_PATH = r"E:\Pycode\Thuchanhbaocao2\model\linear_regression.joblib"
ENCODER_PATH = r"E:\Pycode\Thuchanhbaocao2\model\onehot_encoder.pkl"

lr_model = load(MODEL_PATH)
ohe = load(ENCODER_PATH)

categorical_cols = ['season', 'carrier', 'origin', 'destination', 'year', 'quarter', 'month', 'day']

def preprocess_input(data):
    df = pd.DataFrame([data])
    
    df['departure_date'] = pd.to_datetime(df['departure_date'], errors='coerce')
    df['year'] = df['departure_date'].dt.year
    df['month'] = df['departure_date'].dt.month
    df['day'] = df['departure_date'].dt.day
    df['day_of_week'] = df['departure_date'].dt.dayofweek
    df['quarter'] = df['departure_date'].dt.quarter

    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"
    
    df['season'] = df['month'].apply(get_season)

    X_num = df[['sin_month', 'cos_month', 'sin_dayofweek', 'cos_dayofweek']]

    X_cat = ohe.transform(df[categorical_cols])
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names, index=df.index)

    X_encoded = pd.concat([X_cat_df, X_num], axis=1)
    return X_encoded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        if request.content_type == 'application/json':
            input_data = request.get_json()
        else:
            input_data = request.form.to_dict()


        required = ['carrier', 'departure_date', 'origin', 'destination']
        for field in required:
            if field not in input_data or not input_data[field]:
                return render_error(f"Thiếu trường: {field}")

  
        if input_data['origin'] == input_data['destination']:
            return render_error("Nơi đi và nơi đến không được trùng nhau!")

 
        X = preprocess_input(input_data)
        y_pred = lr_model.predict(X)
        prediction = round(float(y_pred[0]), 4)

        result = {"prediction_gram_co2": prediction}

        if request.content_type == 'application/json':
            return jsonify({**result, "message": "Dự đoán thành công!"})
        else:
            return render_template('index.html', prediction=prediction)

    except Exception as e:
        return render_error(str(e))

def render_error(msg):
    if request.content_type == 'application/json':
        return jsonify({"error": msg}), 400
    else:
        return render_template('index.html', error=msg)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    return predict()

if __name__ == '__main__':
    app.run(debug=True)