
        function predictChurn() {
            document.getElementById("result").innerHTML =
                "<p>Analyzing customer...</p>";
            const button = document.querySelector("button");
            button.disabled = true;
            button.innerText = "Analyzing...";
    const customer = {
        gender: document.getElementById("gender").value,
        SeniorCitizen: Number(document.getElementById("SeniorCitizen").value),
        Partner: document.getElementById("Partner").value,
        Dependents: document.getElementById("Dependents").value,
        tenure: Number(document.getElementById("tenure").value),
        PhoneService: document.getElementById("PhoneService").value,
        MultipleLines: document.getElementById("MultipleLines").value,
        InternetService: document.getElementById("InternetService").value,
        OnlineSecurity: document.getElementById("OnlineSecurity").value,
        OnlineBackup: document.getElementById("OnlineBackup").value,
        DeviceProtection: document.getElementById("DeviceProtection").value,
        TechSupport: document.getElementById("TechSupport").value,
        StreamingTV: document.getElementById("StreamingTV").value,
        StreamingMovies: document.getElementById("StreamingMovies").value,
        Contract: document.getElementById("Contract").value,
        PaperlessBilling: document.getElementById("PaperlessBilling").value,
        PaymentMethod: document.getElementById("PaymentMethod").value,
        MonthlyCharges: Number(document.getElementById("MonthlyCharges").value),
        TotalCharges: Number(document.getElementById("TotalCharges").value)
    };

    if (
    customer.tenure === 0 ||
    customer.MonthlyCharges === 0 ||
    customer.TotalCharges === 0
) {
    alert("Please enter valid Tenure, Monthly Charges, and Total Charges.");
    return;
}
    console.log(customer);
    fetch("/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(customer)
})
.then(response => response.json())
.then(data => {
    console.log(data);
    const button = document.querySelector("button");
    button.disabled = false;
    button.innerText = "Predict Churn";
    const result = document.getElementById("result");
    if (data.error) {
    result.innerHTML =
        "<h2>Prediction Error</h2>" +
        "<p>" + data.error + "</p>";
    return;
}

    result.innerHTML =
    "<h2>Prediction Result</h2>" +
    "<p><strong>Prediction:</strong> " +
    (data.prediction === 1 ? "Churn" : "No Churn") +
    "</p>" +
    "<p><strong>Churn Probability:</strong> " +
(data.churn_probability * 100).toFixed(2) + "%" +
"</p>" +
"<div class='probability-bar'>" +
"<div class='probability-fill' style='width:" +
(data.churn_probability * 100) + "%'></div>" +
"</div>" +
    "<p><strong>Risk Level:</strong> " +
    "<span class='risk-text'>" + data.risk + "</span>" +
    "</p>" +
    "<p><strong>Recommendations:</strong></p>" +
    "<ul>" +
    data.recommendation.map(item => "<li>" + item + "</li>").join("") +
    "</ul>";
    result.className = "result-card " + data.risk.toLowerCase();
})

.catch(error => {
    const button = document.querySelector("button");
    button.disabled = false;
    button.innerText = "Predict Churn";

    const result = document.getElementById("result");

    result.innerHTML =
        "<h2>Connection Error</h2>" +
        "<p>Unable to connect to the prediction server.</p>";
});
}
 