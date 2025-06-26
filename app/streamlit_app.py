import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load(r".\models\model.joblib")

st.title("Customer Churn Prediction")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, step=1.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, step=1.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Prepare input data
input_data = pd.DataFrame({
    "gender": [1 if gender == "Male" else 0],
    "Partner": [1 if partner == "Yes" else 0],
    "Dependents": [1 if dependents == "Yes" else 0],
    "tenure": [tenure],
    "PhoneService": [1 if phone_service == "Yes" else 0],
    "PaperlessBilling": [1 if paperless_billing == "Yes" else 0],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract_Month-to-month": [1 if contract == "Month-to-month" else 0],
    "Contract_One year": [1 if contract == "One year" else 0],
    "Contract_Two year": [1 if contract == "Two year" else 0],
})

# Fill missing columns with 0 (for one-hot encoded columns not in input)
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model.feature_names_in_]

# Predict churn
if st.button("Predict"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    st.write(f"Churn Probability: {proba:.2f}")
    st.success("Likely to Stay" if prediction[0] == 0 else "At Risk of Churning")
