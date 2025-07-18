
# You would typically run this in your local environment or a dedicated Streamlit hosting platform
# First, make sure you have streamlit installed: pip install streamlit pandas scikit-learn xgboost

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # In case we need to handle any missing values in new data
from datetime import datetime # To handle Date of Joining

# Load the trained model and scaler
try:
    with open('best_gb_model.pkl', 'rb') as f:
        best_gb_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please make sure 'best_gb_model.pkl' and 'scaler.pkl' are in the same directory as your Streamlit app script.")
    st.stop() # Stop the app if files are not found

# Define the expected features in the correct order for the model after preprocessing
# This list should exactly match the columns in X_train that the model was trained on.
# Based on the notebook: Designation, Resource Allocation, Mental Fatigue Score,
# Gender_Female, Gender_Male, Company Type_Product, Company Type_Service,
# WFH Setup Available_No, WFH Setup Available_Yes, Tenure
expected_features = ['Designation', 'Resource Allocation', 'Mental Fatigue Score',
                     'Gender_Female', 'Gender_Male', 'Company Type_Product',
                     'Company Type_Service', 'WFH Setup Available_No',
                     'WFH Setup Available_Yes', 'Tenure']


# Streamlit App Title
st.title("Employee Burnout Prediction App")

st.write("Enter employee details to predict their burnout rate.")

# User Input Section
st.header("Employee Details")

# Input fields for features
designation = st.slider("Designation (Seniority Level)", 0, 5, 2)
resource_allocation = st.slider("Resource Allocation (Hours per day)", 1, 10, 5)
mental_fatigue_score = st.slider("Mental Fatigue Score (0-10)", 0.0, 10.0, 5.0)
gender = st.selectbox("Gender", ['Female', 'Male'])
company_type = st.selectbox("Company Type", ['Service', 'Product'])
wfh_setup_available = st.selectbox("WFH Setup Available", ['Yes', 'No'])
date_of_joining = st.date_input("Date of Joining", datetime(2008, 9, 30)) # Set a default date

# Calculate Tenure based on the date of joining and a fixed snapshot date
# It's important to use the same logic as in the training notebook.
# Let's use the snapshot_date derived from the max date in the training data + 30 days
# (which was approximately '2009-01-01' based on the sample output)
# In a real app, you might want to save and load this snapshot_date or calculate relative tenure
snapshot_date = pd.to_datetime('2009-01-01') # Use the same logic as in training
tenure = (snapshot_date - pd.to_datetime(date_of_joining)).days / 365.25


# Create a dictionary from user inputs
input_data = {
    'Designation': designation,
    'Resource Allocation': resource_allocation,
    'Mental Fatigue Score': mental_fatigue_score,
    'Gender': gender,
    'Company Type': company_type,
    'WFH Setup Available': wfh_setup_available,
    'Tenure': tenure
}

# Convert input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])

# --- Preprocessing the input data ---

# Apply One-Hot Encoding (manual implementation based on training columns)
# In a real scenario, it's best to save and load the fitted OneHotEncoder object.
# This manual approach requires careful mapping to the expected features.
processed_input = pd.DataFrame(0, index=input_df.index, columns=expected_features)

# Populate the processed_input DataFrame based on user inputs
processed_input['Designation'] = input_df['Designation']
processed_input['Resource Allocation'] = input_df['Resource Allocation']
processed_input['Mental Fatigue Score'] = input_df['Mental Fatigue Score']
processed_input['Tenure'] = input_df['Tenure']

# Handle categorical features with one-hot encoding logic
if input_df['Gender'].iloc[0] == 'Female':
    processed_input['Gender_Female'] = 1
else:
    processed_input['Gender_Male'] = 1

if input_df['Company Type'].iloc[0] == 'Product':
    processed_input['Company Type_Product'] = 1
else:
    processed_input['Company Type_Service'] = 1

if input_df['WFH Setup Available'].iloc[0] == 'No':
    processed_input['WFH Setup Available_No'] = 1
else:
    processed_input['WFH Setup Available_Yes'] = 1

# Ensure columns are in the correct order before scaling and prediction
processed_input = processed_input[expected_features]

# Scale the numerical features using the loaded scaler
# Identify numerical columns based on expected_features (those not OHE)
numerical_cols_for_scaling = ['Designation', 'Resource Allocation', 'Mental Fatigue Score', 'Tenure']
processed_input[numerical_cols_for_scaling] = scaler.transform(processed_input[numerical_cols_for_scaling])


# --- Prediction ---
if st.button("Predict Burnout Rate"):
    # Make prediction
    predicted_burnout_rate = best_gb_model.predict(processed_input)

    # Display the prediction
    st.header("Prediction")
    st.success(f"Predicted Employee Burnout Rate: {predicted_burnout_rate[0]:.4f}")

# Optional: Add an expander to show the processed input data
with st.expander("See processed input data for prediction"):
    st.write(processed_input)

# Note: To run this Streamlit app:
# 1. Save the code above as a Python file (e.g., app.py).
# 2. Make sure 'best_gb_model.pkl' and 'scaler.pkl' are in the same directory.
# 3. Open your terminal or command prompt.
# 4. Navigate to that directory.
# 5. Run the command: streamlit run app.py
