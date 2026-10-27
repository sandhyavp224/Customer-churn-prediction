
# Customer Churn Prediction


# Project Overview

Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to leave a service.

The application analyzes customer information, predicts churn probability, classifies customers into risk levels, and provides rule-based retention recommendations.


## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- HTML
- CSS
- JavaScript
- Git and GitHub


## Features

- Customer churn prediction using Logistic Regression.
- Churn probability estimation.
- Customer risk classification: Low, Medium, and High.
- Automated, rule-based retention recommendations.
- Interactive web interface for entering customer details.
- Flask API for connecting the frontend to the ML model.
- Data preprocessing using a Scikit-learn pipeline.
- Probability visualization using a progress bar.


## Dataset

This project uses the Telco Customer Churn dataset.

The dataset contains customer information, including:
- Customer tenure
- Contract type
- Internet and phone services
- Payment method
- Monthly charges
- Total charges
- Churn status

The target variable is `Churn`:
- `Yes` — The customer left the service.
- `No` — The customer stayed with the service.

Data preprocessing includes converting `TotalCharges` to numeric format, handling missing values, and separating the features from the target variable.


## Model Training and Evaluation

The project uses Logistic Regression as the primary classification model.

The workflow includes:
1. Splitting the dataset into training and testing sets.
2. Preprocessing numerical and categorical features.
3. Training the machine learning model.
4. Predicting customer churn on unseen test data.
5. Evaluating model performance using classification metrics.

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

A Scikit-learn Pipeline combines preprocessing and model training to ensure consistent transformations during prediction.


## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Customer-churn-prediction
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask joblib requests
```

### 3. Run the Flask application

```bash
python app.py
```

### 4. Open the application

Open your browser and visit:

http://127.0.0.1:5000/



## Project Structure

```text
Customer-churn-prediction/
│
├── app.py
├── churn_pipeline.pkl
├── test_api.py
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

## Future Improvements

- Deploy the application so users can access it online.
- Improve churn prediction performance through model comparison and tuning.
- Add an interactive customer-risk dashboard.
- Enhance retention recommendations using customer behavior and business rules.
- Add input validation and automated testing.
