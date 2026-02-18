from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Helper: try absolute path, then project locations (same filename, ./models/, ./data/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_pickle_anywhere(abs_path, desc):
    if os.path.exists(abs_path):
        return pickle.load(open(abs_path, "rb"))
    fname = os.path.basename(abs_path)
    candidates = [
        os.path.join(BASE_DIR, fname),
        os.path.join(BASE_DIR, "models", fname),
        os.path.join(BASE_DIR, "data", fname),
    ]
    for p in candidates:
        if os.path.exists(p):
            return pickle.load(open(p, "rb"))
    raise FileNotFoundError(f"{desc} not found. Tried: {abs_path} and {candidates}")

# Attempt to load pickles but capture any load error so the app can show a helpful message
load_error = None
try:
    model = _load_pickle_anywhere(r"C:\\Users\\kanal\\OneDrive\\Desktop\\rain\\rainfall.pkl", "Model")
    scaler = _load_pickle_anywhere(r"C:\\Users\\kanal\\OneDrive\\Desktop\\rain\\scaler.pkl", "Scaler")
    encoders = _load_pickle_anywhere(r"C:\\Users\\kanal\\OneDrive\\Desktop\\rain\\encoder.pkl", "Encoders")
except Exception as e:
    load_error = str(e)
    model = scaler = encoders = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # If pickles failed to load at startup, return helpful error instead of crashing
        if load_error:
            return f"Model load error: {load_error}"
        if model is None or scaler is None:
            return "Model or scaler not loaded. Check server logs for details."

        input_data = request.form.to_dict()

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # 🔹 Convert numeric columns to float
        numeric_cols = [
            'MinTemp',
            'MaxTemp',
            'Rainfall',
            'Humidity9am',
            'Humidity3pm',
            'Pressure9am',
            'Pressure3pm'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # 🔹 Encode categorical columns (skip if no encoders)
        local_encoders = encoders or {}
        for col in local_encoders:
            if col in df.columns:
                try:
                    df[col] = local_encoders[col].transform(df[col])
                except Exception as e:
                    return f"Encoding Error for {col}: {e}"

        # 🔹 Scale
        scaled_data = scaler.transform(df)

        # 🔹 Predict
        prediction = model.predict(scaled_data)[0]

        if prediction == 1:
            return render_template("chance.html")
        else:
            return render_template("noChance.html")

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)