import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved CatBoost model
try:
    model = joblib.load('rf_model.pkl')
    st.success("CatBoost model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'churn_prediction_model.pkl' not found. Please make sure the file is in the same directory.")
    st.stop()

# Define columns used during training
model_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
    'MonthlyCharges', 'TotalCharges', 'MultipleLines_No',
    'MultipleLines_Yes', 'InternetService_DSL',
    'InternetService_Fiber optic', 'InternetService_No',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Load customer data for recommendations
X = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
y = X['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Target variable for churn status

# UI Elements
st.title("Customer Churn Prediction and Recommendation System")

# User input fields
gender = st.selectbox("Gender", options=['Male', 'Female'], index=1)
tenure = st.slider("Tenure", min_value=0, max_value=100, value=24)
monthly_charges = st.number_input("Monthly Charges", value=70.0)
total_charges = st.number_input("Total Charges", value=840.0)
paperless_billing = st.selectbox("Paperless Billing", options=['Yes', 'No'], index=0)

# Default values for other features
default_values = {
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaymentMethod': 'Electronic check',
}

# Assemble user inputs
input_data = {
    'gender': gender,
    'SeniorCitizen': default_values['SeniorCitizen'],
    'Partner': default_values['Partner'],
    'Dependents': default_values['Dependents'],
    'tenure': tenure,
    'PhoneService': default_values['PhoneService'],
    'MultipleLines': default_values['MultipleLines'],
    'InternetService': default_values['InternetService'],
    'OnlineSecurity': default_values['OnlineSecurity'],
    'OnlineBackup': default_values['OnlineBackup'],
    'DeviceProtection': default_values['DeviceProtection'],
    'TechSupport': default_values['TechSupport'],
    'StreamingTV': default_values['StreamingTV'],
    'StreamingMovies': default_values['StreamingMovies'],
    'PaperlessBilling': paperless_billing,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': default_values['Contract'],
    'PaymentMethod': default_values['PaymentMethod'],
}

input_df = pd.DataFrame([input_data])
input_encoded = pd.get_dummies(input_df)

# Add missing columns with 0s for alignment with model_columns
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns to match model requirements
input_encoded = input_encoded[model_columns]

# Define function to generate recommendations
def generate_recommendations(user_data):
    # Filter for churned customers
    churned_customers = X[X['Churn'] == 'Yes']

    # Average thresholds based on churned customers
    avg_tenure = np.mean(churned_customers['tenure'])
    avg_monthly_charges = np.mean(churned_customers['MonthlyCharges'])
    avg_total_charges = np.mean(churned_customers['TotalCharges'].replace(' ', '0').astype(float))
    majority_gender = churned_customers['gender'].mode()[0]
    recommendations = {}

    # Example recommendation conditions
    if user_data['Contract'] == 'Month-to-month':
        recommendations['Contract'] = "Consider offering a discounted long-term contract."
    if user_data['PaperlessBilling'] == 'No':
        recommendations['PaperlessBilling'] = "Suggest enabling paperless billing for ease of access."
    if user_data['InternetService'] == 'DSL':
        recommendations['InternetService'] = "Consider upgrading to fiber optic for better service."
    if user_data['OnlineSecurity'] == 'No':
        recommendations['OnlineSecurity'] = "Promote online security services as an add-on."
    if user_data['OnlineBackup'] =='No':
        recommendations['OnlineBackup'] = "Include online backup service."
    if user_data['DeviceProtection'] == 0:
        recommendations['DeviceProtection'] = "Add device protection plan."
    if user_data['StreamingTV'] == 0 :
        if user_data['StreamingMovies'] == 0:
            recommendations['StreamingTV'] = "Suggest TV & Movies streaming services at discounted prices."
    # Recommendations based on tenure
    if user_data['tenure'] < avg_tenure:
        recommendations['Tenure'] = "Offer loyalty incentives to increase customer tenure"

    # Recommendations based on monthly charges
    if user_data['MonthlyCharges'] > avg_monthly_charges:
        recommendations['MonthlyCharges'] = "Provide a customized discount to reduce monthly charges"
    
    # Recommendations based on total charges
    if user_data['TotalCharges'] < avg_total_charges:
        recommendations['TotalCharges'] = "Consider suggesting a higher-tier plan to enhance service experience"

    # Gender-based recommendations
    if user_data['gender'] != majority_gender:
        recommendations['Gender'] = f"Consider targeted marketing strategies appealing to {user_data['gender']} customers"


    return recommendations

# Churn prediction and recommendations
if st.button("Predict Churn and Get Recommendations"):
    # Churn prediction and probability
    prediction = model.predict(input_encoded)
    churn_prob = model.predict_proba(input_encoded)[0][1]

    st.write(f"**Predicted Churn:** {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"**Churn Probability:** {churn_prob:.2%}")

    # Generate personalized recommendations
    recommendations = generate_recommendations(input_data)

    # Organized display of recommendations
    with st.expander("🔍 Personalized Recommendations", expanded=True):
        st.write("Based on your current profile, we suggest the following actions:")
        for key, recommendation in recommendations.items():
            st.write(f"- **{recommendation}**")

    # Calculate cosine similarity between input and customer profiles
    customer_profiles_encoded = pd.get_dummies(X)
    for col in model_columns:
        if col not in customer_profiles_encoded.columns:
            customer_profiles_encoded[col] = 0
    customer_profiles_encoded = customer_profiles_encoded[model_columns]

    similarities = cosine_similarity(input_encoded, customer_profiles_encoded)
    similar_customer_indices = similarities.argsort()[0][-5:]

    # Display similar customer information
    with st.expander("📋 Insights from Similar Customers"):
        st.write("Here are profiles of customers similar to yours:")
        for idx in similar_customer_indices:
            st.write(f"- **Similar Customer {idx + 1}:**")
            st.write(f"  - Contract: **{X.iloc[idx]['Contract']}**")
            st.write(f"  - Monthly Charges: **${X.iloc[idx]['MonthlyCharges']:.2f}**")

    # Additional retention strategies
    with st.expander("📈 Additional Retention Strategies"):
        st.write("To enhance retention, consider implementing these strategies:")
        st.write("- **Offer discounts on long-term contracts** to incentivize loyalty.")
        st.write("- **Promote online security or device protection services** to add more value to the subscription.")

