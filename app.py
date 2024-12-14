from flask import Flask, request, render_template, jsonify
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
        # Get input data from the form
        features = [float(x) for x in request.form.values()]  # Convert input to floats
        features = np.array(features).reshape(1, -1)

        # Normalize the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]

        # Render the result on the same page
        return render_template(
            "index.html",
            prediction_text=f"Prediction: {'Disease Present' if prediction[0] == 1 else 'No Disease'}",
            probability_text=f"Probability: {round(probability * 100, 2)}%"
        )
    except Exception as e:
        return render_template("index.html", prediction_text="Error: Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)

