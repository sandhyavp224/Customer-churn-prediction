import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score , recall_score , f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib 
df = pd.read_csv("data/telco_churn.csv")


# replace the empty string with nan text
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

#drop the row with nan values
df = df.dropna()

# convert the column to float
df["TotalCharges"] = df["TotalCharges"].astype(float)
#display a first 5 rows
# print(df.head())
# # display a shape of dataset
# print("Shape of Dataset:",df.shape)

# #Column namesnex
# print(df.columns)

# #knowing the data types of each column
# print(df.dtypes)+

# #dataset information
# df.info()

# #Statistical summary of the dataset
# print(df.describe())


# #check the missing values
# print("\n Missing values:")
# print(df.isnull().sum())

# #check for duplicates rows
# print("\n Duplicate rows:", df.duplicated().sum())

# #check unique
# print("\nUnique Values:")
# print(df.nunique())

# #==============================

# #Preprocessing the data

# #==============================

# df = df.drop("customerID", axis = 1)
# print(df.head())
# print(df.dtypes)
# print(type(df["TotalCharges"][0]))

# print(df[df["TotalCharges"] == " "])
# print(df["TotalCharges"].dtype)
# print(df["Churn"].value_counts())

# plt.figure(figsize=(6,4))
# sns.countplot(x="Churn", data = df)
# plt.title("Customer Churn Distribution")
# plt.show()
# print(df["Churn"].value_counts())

# plt.figure(figsize = (6,4))
# sns.countplot(x="gender" , hue = "Churn", data = df)
# plt.title("Gender vs Customer Churn")
# plt.show()

# contract_churn = pd.crosstab(
#     df["Contract"],
#     df["Churn"],
#     normalize="index"
# ) * 100

# print(contract_churn)
# plt.figure(figsize=(8,5))
# sns.histplot(data = df, x="tenure", hue ="Churn", bins=30, kde=True) 
# plt.title("Tenure Distribution by Churn")
# plt.xlabel("Tenure")
# plt.ylabel("Number of Customers")
# plt.show()

# plt.figure(figsize=(8,5))
# sns.histplot(
#     data = df,
#     x = "tenure",
#     hue = "Churn",
#     bins = 30 ,
#     kde=True
    
# )
# plt.title("Tenure Distribution bu Churn")
# plt.xlabel("Tenure (Months)")
# plt.ylabel("Number of Customers")
# plt.show()
# plt.figure(figsize=(8, 5))

# sns.histplot(
#     data=df,
#     x="MonthlyCharges",
#     hue="Churn",
#     bins=30,
#     kde=True
# )

# plt.title("Monthly Charges Distribution by Churn")
# plt.xlabel("Monthly Charges")
# plt.ylabel("Number of Customers")

# plt.show()
print(df.groupby("Churn")["MonthlyCharges"].mean())
churn_payment = pd.crosstab(
    df['PaymentMethod'] ,
    df['Churn'],
    normalize = "index"
) * 100
print(churn_payment)

#Remove customerid it is only for identifier 
df = df.drop("customerID" , axis = 1)

X = df.drop("Churn" , axis = 1 )
y = df['Churn'] # churn is the taget so its 19 columns

print("X shape:", X.shape)
print("y shape:", y.shape)
print(df.shape)
print(df.columns.tolist())

categorical_cols = X.select_dtypes(include="object").columns

print(categorical_cols)
X_encoded = pd.get_dummies(X, drop_first=True)

print("Before encoding:", X.shape)
print("After encoding:", X_encoded.shape)
print(X_encoded.head())
y = y.map({"No": 0, "Yes": 1})

# print(y.head())
# print(y.value_counts())


#==============================
#TRAINING AND TESTING MODEL 
#==============================

X_train , X_test , y_train , y_test = train_test_split(
    X_encoded , 
    y,
    test_size = 0.2,
    random_state=42,
    stratify = y
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)

model = LogisticRegression()
model.fit(X_train_scaled, y_train) # Train the model using training data
print("Model training completed!")

y_pred = model.predict(X_test_scaled)

print(y_pred)
#############################################
# Training Model 

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred  = model.predict(X_test)


accuracy = accuracy_score(y_test , y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test , y_pred)
recall = recall_score(y_test , y_pred)
f1 = f1_score(y_test , y_pred)

print("Precision:" , precision)
print("Recall:" , recall)
print("F1 - Score:", f1)
cm = confusion_matrix(y_test , y_pred)
print(cm)


# RANDOM FOREST CLASSIFIER  MODEL
rf_model = RandomForestClassifier(random_state = 42)
# creared
###TRANING IT
rf_model.fit(X_train, y_train) # teach model

#STEP 3 : PREDITCION
rf_pred = rf_model.predict(X_test)

# STEP 4 : CALCULATE ACCURACY
rf_accuracy = accuracy_score(y_test , rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test , rf_pred)
rf_f1 = f1_score(y_test , rf_pred)
rf_cm = confusion_matrix(y_test , rf_pred)

print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1-Score:", rf_f1)
print(rf_cm)


dt_model = DecisionTreeClassifier(random_state = 43)
dt_model.fit(X_train , y_train)
dt_pred = dt_model.predict(X_test)

y_prob = model.predict_proba(X_test)

print(y_prob[:5])
"""
X_test
  ↓
Trained Decision Tree
  ↓
Predicted churn/stay

step 2 : dt_pred  ← model's prediction
         y_test   ← actual answer 
 """

coefficients = model.coef_[0]
print(coefficients)
print(X_encoded.columns)
print(model.coef_[0])


# Testing new customer
numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]
categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)
# #"num"
#    ↓
# numerical columns
#    ↓
# "passthrough"
#    ↓
# keep them

# "cat"
#    ↓
# categorical columns
#    ↓
# OneHotEncoder
#    ↓
# convert categories → numbers
# handle_unknown="ignore"

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X_train.shape)
print(X_test.shape)
print(X_train.columns)
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "churn_pipeline.pkl")

pipeline_pred = pipeline.predict(X_test)
print(pipeline_pred[:10])
pipeline_prob = pipeline.predict_proba(X_test)
print(pipeline_prob[:5])
churn_probability = pipeline_prob[0][1]
if churn_probability < 0.30:
  risk = "Low"
elif churn_probability < 0.60:
  risk = "Medium"
else:
  risk = "High"
churn_probabilities = pipeline_prob[:, 1]


churn_probabilities = pipeline_prob[:, 1]
print(churn_probabilities[:5])
for i , probability in enumerate(churn_probabilities):
    if probability < 0.30:
        risk = "Low"
    elif probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    print(f"Customer {i + 1} -> Chrun Probability: {probability:.2f} -> Risk: {risk}")
print(repr(churn_probabilities[395]))
for i, probability in enumerate(churn_probabilities):
    if 0.59 < probability < 0.60:
        print(f"Customer {i + 1}: {probability!r}")



feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

print(feature_names)
coefficients = pipeline.named_steps["model"].coef_[0]

print(coefficients)
customer = X_test.iloc[[0]]
transformed_customer = pipeline.named_steps["preprocessor"].transform(customer)

print(transformed_customer.toarray() if hasattr(transformed_customer, "toarray") else transformed_customer)
print(type(customer))
print(customer.shape)
# -----------------------------------
# RETENTION RECOMMENDATION
# -----------------------------------

def retention_recommendation(risk):
    if risk == "High":
        return "Offer a suitable retention benefit and contact the customer."
    elif risk == "Medium":
        return "Monitor the customer and consider a retention offer."
    else:
        return "No immediate action needed."


# -----------------------------------
# NEW CUSTOMER
# -----------------------------------

new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0,
    "TotalCharges": 400.0
}])


# -----------------------------------
# PREDICTION
# -----------------------------------

prediction = pipeline.predict(new_customer)[0]

probability = pipeline.predict_proba(new_customer)[0][1]


# -----------------------------------
# RISK LEVEL
# -----------------------------------

if probability < 0.30:
    risk = "Low"
elif probability < 0.60:
    risk = "Medium"
else:
    risk = "High"


# -----------------------------------
# RECOMMENDATION
# -----------------------------------

recommendation = retention_recommendation(risk)


# -----------------------------------
# FINAL OUTPUT
# -----------------------------------

if prediction == 1:
    prediction_text = "Churn"
else:
    prediction_text = "Stay"

print(f"Prediction: {prediction_text}")
print(f"Churn Probability: {probability:.2%}")
print(f"Risk Level: {risk}")
print(f"Recommendation: {recommendation}")

def smart_recommendation(customer, risk):
    
    recommendations = []
    
    if customer["Contract"].iloc[0] == "Month-to-month":
        recommendations.append(
            "Consider offering a longer-term contract benefit"
        )
    
    if customer["PaymentMethod"].iloc[0] == "Electronic check":
        recommendations.append(
            "Consider encouraging automatic payment methods"
        )
    
    if customer["tenure"].iloc[0] < 12:
        recommendations.append(
            "Consider an early-tenure retention offer."
        )
    
    if risk == "High":
        recommendations.append(
            "Prioritize this customer for retention outreach"
        )
    
    return recommendations
smart_recommendations = smart_recommendation(new_customer, risk)
print("Smart Recommendations:")

for recommnedation in smart_recommendations:
    print("-", recommnedation)
    