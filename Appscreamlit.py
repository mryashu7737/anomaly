import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('model/model.pkl')  # Ensure this path is correct

# Load the data statistics
data_path = "full_user_anomaly_detection_data.csv"  # Ensure this path is correct
df = pd.read_csv(data_path)
data_stats = df.describe().T

# Function to identify anomalies and provide suggestions
def analyze_anomaly(new_data):
    reasons = []
    suggestions = []
    
    for column in data_stats.index:
        mean_val = data_stats.loc[column, 'mean']
        std_dev = data_stats.loc[column, 'std']
        
        # Calculate Z-score
        z_score = (new_data[column] - mean_val) / std_dev
        
        if abs(z_score) > 3:  # If Z-score > 3, it's considered an anomaly
            anomaly_type = 'high' if z_score > 3 else 'low'
            reasons.append(f"{column} is too {anomaly_type} (Z-score: {z_score:.2f})")
            
            # Suggestion based on typical values
            if anomaly_type == 'high':
                suggestions.append(f"Reduce {column} closer to the average of {mean_val:.2f}.")
            else:
                suggestions.append(f"Increase {column} closer to the average of {mean_val:.2f}.")
                
    return reasons, suggestions

# Streamlit UI
st.title("Anomaly Detection for Financial Records")

st.header("Enter your financial details:")

# Collecting user input
income = st.number_input("Income", min_value=0.0, value=0.0)
deductions = st.number_input("Deductions", min_value=0.0, value=0.0)
filing_status = st.selectbox("Filing Status", options=[1, 2, 3], format_func=lambda x: ["Single", "Married", "Head of Household"][x-1])
home_loan_interest = st.number_input("Home Loan Interest", min_value=0.0, value=0.0)
education_loan_interest = st.number_input("Education Loan Interest", min_value=0.0, value=0.0)
charitable_donations = st.number_input("Charitable Donations", min_value=0.0, value=0.0)

# Creating a DataFrame for the new input
new_user_data = pd.DataFrame([[income, deductions, filing_status, home_loan_interest, education_loan_interest, charitable_donations]],
                             columns=["Income", "Deductions", "Filing_Status", "Home_Loan_Interest", "Education_Loan_Interest", "Charitable_Donations"])

if st.button("Check for Anomalies"):
    # Predict whether the new data is an anomaly or not
    prediction = model.predict(new_user_data)
    
    if prediction[0] == -1:
        st.error("The entered record is an ANOMALY (Outlier).")
        reasons, suggestions = analyze_anomaly(new_user_data.iloc[0])
        
        # Display detailed reasons and suggestions
        st.subheader("Reasons for being classified as an anomaly:")
        for reason in reasons:
            st.write(f"- {reason}")
        
        st.subheader("Suggestions to resolve the anomaly:")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
    else:
        st.success("The entered record is NORMAL.")
