
import streamlit as st
import pandas as pd
import pickle
import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder # Not directly used for loading/encoding
# from sklearn.impute import SimpleImputer # Not directly used in the app
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
# IMPORTANT: This snapshot_date MUST be the exact same one used during model training.
# Based on our analysis, the snapshot_date was derived from max_date + 30 days.
# If it was precisely '2009-01-01' from your training, keep it. Otherwise, use the exact one.
# For consistency with the last calculated snapshot_date based on max_date_train/test in your notebook,
snapshot_date = pd.to_datetime('2009-01-01')
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

# Create a DataFrame with all expected columns, initialized to 0
processed_input = pd.DataFrame(0, index=input_df.index, columns=expected_features)

# Populate numerical and engineered features
processed_input['Designation'] = input_df['Designation']
processed_input['Resource Allocation'] = input_df['Resource Allocation']
processed_input['Mental Fatigue Score'] = input_df['Mental Fatigue Score']
processed_input['Tenure'] = input_df['Tenure']

# Handle categorical features with manual one-hot encoding logic
# Ensure these map correctly to your expected_features
if input_df['Gender'].iloc[0] == 'Female':
    processed_input['Gender_Female'] = 1
else: # Male
    processed_input['Gender_Male'] = 1

if input_df['Company Type'].iloc[0] == 'Product':
    processed_input['Company Type_Product'] = 1
else: # Service
    processed_input['Company Type_Service'] = 1

if input_df['WFH Setup Available'].iloc[0] == 'No':
    processed_input['WFH Setup Available_No'] = 1
else: # Yes
    processed_input['WFH Setup Available_Yes'] = 1

# Ensure columns are in the exact order the model expects
# This is crucial for correct prediction
processed_input = processed_input[expected_features]

# Scale ALL features using the loaded scaler
# The scaler was fit on all features (including one-hot encoded ones) during training
processed_input_scaled = scaler.transform(processed_input)


# --- Prediction ---
if st.button("Predict Burnout Rate"):
    # Make prediction using the scaled input
    predicted_burnout_rate = best_gb_model.predict(processed_input_scaled)

    # Display the prediction
    st.header("Prediction")
    st.success(f"Predicted Employee Burnout Rate: {predicted_burnout_rate[0]:.4f}")

# Optional: Add an expander to show the processed input data
with st.expander("See processed input data for prediction"):
    st.write(pd.DataFrame(processed_input_scaled, columns=expected_features)) # Show scaled data
