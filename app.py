from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # Serve the HTML form
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Log received form data for debugging
        print("Form Data:", request.form)

        # Collect inputs from form
        feature_names = [
            "age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
        ]
        features = [float(request.form[feature]) for feature in feature_names]

        # Log the collected features for debugging
        print("Collected Features:", features)

        # Convert input to a DataFrame with column names
        features_df = pd.DataFrame([features], columns=feature_names)

        # Normalize and predict
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]

        # Render results
        return render_template(
            "index.html",
            prediction_text=f"Prediction: {'Disease Present' if prediction[0] == 1 else 'No Disease'}",
            probability_text=f"Probability: {round(probability * 100, 2)}%"
        )
    except Exception as e:
        print("Error:", str(e))  # Log the error for debugging
        return render_template("index.html", prediction_text="Error: Invalid Input")



if __name__ == "__main__":
    app.run(debug=True)

