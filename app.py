from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd


app = Flask(__name__)
pipeline = joblib.load("churn_pipeline.pkl")

@app.route("/")

def home():
    return render_template("index.html")

def retention_recommendation(customer, risk):

    recommendations = []

    if customer["Contract"].iloc[0] == "Month-to-month":
        recommendations.append(
            "Consider offering a longer-term contract benefit."
        )

    if customer["PaymentMethod"].iloc[0] == "Electronic check":
        recommendations.append(
            "Consider encouraging automatic payment methods."
        )

    if customer["tenure"].iloc[0] < 12:
        recommendations.append(
            "Consider an early-tenure retention offer."
        )

    if risk == "High":
        recommendations.append(
            "Prioritize this customer for retention outreach."
        )

    if risk == "Medium":
        recommendations.append(
            "Monitor this customer and consider a retention offer."
        )

    if risk == "Low":
        recommendations.append(
            "No immediate retention action is needed."
        )

    return recommendations
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if not data:
        return jsonify({"error": "No customer data received"}), 400
    
    
    customer = pd.DataFrame([data])
    

    try:
    # Make prediction and get probability of churn
        prediction = pipeline.predict(customer)
        probability = pipeline.predict_proba(customer)[0][1]

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    # Convert churn probability into a risk level
    if probability < 0.30:
        risk = "Low"
    elif probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"
        
    # Generate retention recommendations based on customer details and risk
    recommendation = retention_recommendation(customer, risk)
    return jsonify({
    "prediction": int(prediction[0]),
    "churn_probability": float(probability),
    "risk": risk,
    "recommendation": recommendation
})


if __name__ == "__main__":
    app.run(debug=True)
    
