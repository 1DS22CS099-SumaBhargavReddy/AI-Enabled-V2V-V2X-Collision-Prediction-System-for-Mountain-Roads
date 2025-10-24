from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("../models/baseline_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # expect JSON array of vehicle features
    df = pd.DataFrame(data)
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)
    return jsonify(pred.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
