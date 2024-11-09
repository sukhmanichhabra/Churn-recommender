import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved Random Forest model
try:
    model = joblib.load('catboost_model.pkl')
    st.success("Random Forest model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'rf_model.pkl' not found. Please make sure the file is in the same directory.")
    st.stop()

# Define columns used during training
model_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
       'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
       'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
       'PhoneService_Yes', 'MultipleLines_No',
       'MultipleLines_No phone service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No internet service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
       'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

# Load customer data for recommendations
X = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
y = X['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# UI Elements
st.title("Customer Churn Prediction and Recommendation System")

# Collect user inputs for each feature
gender = st.selectbox("Gender", options=['Male', 'Female'], index=1)
senior_citizen = st.selectbox("Senior Citizen", options=[0, 1], index=0)
partner = st.selectbox("Partner", options=['Yes', 'No'], index=1)
dependents = st.selectbox("Dependents", options=['Yes', 'No'], index=1)
tenure = st.slider("Tenure", min_value=0, max_value=100, value=24)
phone_service = st.selectbox("Phone Service", options=['Yes', 'No'], index=0)
multiple_lines = st.selectbox("Multiple Lines", options=['Yes', 'No'], index=1)
internet_service = st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No'], index=0)
online_security = st.selectbox("Online Security", options=['Yes', 'No'], index=1)
online_backup = st.selectbox("Online Backup", options=['Yes', 'No'], index=1)
device_protection = st.selectbox("Device Protection", options=['Yes', 'No'], index=1)
tech_support = st.selectbox("Tech Support", options=['Yes', 'No'], index=1)
streaming_tv = st.selectbox("Streaming TV", options=['Yes', 'No'], index=1)
streaming_movies = st.selectbox("Streaming Movies", options=['Yes', 'No'], index=1)
paperless_billing = st.selectbox("Paperless Billing", options=['Yes', 'No'], index=0)
monthly_charges = st.number_input("Monthly Charges", value=70.0)
total_charges = st.number_input("Total Charges", value=840.0)
contract = st.selectbox("Contract", options=['Month-to-month', 'One year', 'Two year'], index=0)
payment_method = st.selectbox(
    "Payment Method",
    options=[
        'Bank transfer (automatic)', 'Credit card (automatic)',
        'Electronic check', 'Mailed check'
    ],
    index=2
)

# Assemble user inputs into a dictionary
input_data = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'PaperlessBilling': paperless_billing,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract,
    'PaymentMethod': payment_method,
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])

# One-hot encode the user input to match the training format
input_encoded = pd.get_dummies(input_df)

# Ensure all model columns are in the encoded input
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0  # Add missing columns with a value of 0

# Reorder columns to match the model's expected input format
input_encoded = input_encoded[model_columns]

# Define function to generate recommendations
def generate_recommendations(user_data):
    churned_customers = X[X['Churn'] == 'Yes']
    avg_tenure = np.mean(churned_customers['tenure'])
    avg_monthly_charges = np.mean(churned_customers['MonthlyCharges'])
    avg_total_charges = np.mean(churned_customers['TotalCharges'].replace(' ', '0').astype(float))
    majority_gender = churned_customers['gender'].mode()[0]
    recommendations = {}

    if user_data['Contract'] == 'Month-to-month':
        recommendations['Contract'] = "Consider offering a discounted long-term contract."
    if user_data['PaperlessBilling'] == 'No':
        recommendations['PaperlessBilling'] = "Suggest enabling paperless billing for ease of access."
    if user_data['InternetService'] == 'DSL':
        recommendations['InternetService'] = "Consider upgrading to fiber optic for better service."
    if user_data['OnlineSecurity'] == 'No':
        recommendations['OnlineSecurity'] = "Promote online security services as an add-on."
    if user_data['OnlineBackup'] == 'No':
        recommendations['OnlineBackup'] = "Include online backup service."
    if user_data['DeviceProtection'] == 'No':
        recommendations['DeviceProtection'] = "Add device protection plan."
    if user_data['StreamingTV'] == 'No' and user_data['StreamingMovies'] == 'No':
        recommendations['StreamingTV'] = "Suggest TV & Movies streaming services at discounted prices."
    if user_data['tenure'] < avg_tenure:
        recommendations['Tenure'] = "Offer loyalty incentives to increase customer tenure"
    if user_data['MonthlyCharges'] > avg_monthly_charges:
        recommendations['MonthlyCharges'] = "Provide a customized discount to reduce monthly charges"
    if user_data['TotalCharges'] < avg_total_charges:
        recommendations['TotalCharges'] = "Consider suggesting a higher-tier plan to enhance service experience"
    if user_data['gender'] != majority_gender:
        recommendations['Gender'] = f"Consider targeted marketing strategies appealing to {user_data['gender']} customers"
    
    return recommendations

# Churn prediction and recommendations
if st.button("Predict Churn and Get Recommendations"):
    st.write(input_encoded)
    prediction = model.predict(input_encoded)
    churn_prob = model.predict_proba(input_encoded)[0][1]

    # st.write(f"**Predicted Churn:** {'Yes' if churn_prob > 50 else 'No'}")
    st.write(f"**Churn Probability:** {churn_prob:.2%}")

    recommendations = generate_recommendations(input_data)

    with st.expander("🔍 Personalized Recommendations", expanded=True):
        st.write("Based on your current profile, we suggest the following actions:")
        for key, recommendation in recommendations.items():
            st.write(f"- **{recommendation}**")

    customer_profiles_encoded = pd.get_dummies(X)
    for col in model_columns:
        if col not in customer_profiles_encoded.columns:
            customer_profiles_encoded[col] = 0
    customer_profiles_encoded = customer_profiles_encoded[model_columns]

    similarities = cosine_similarity(input_encoded, customer_profiles_encoded)
    similar_customer_indices = similarities.argsort()[0][-5:]

    with st.expander("📋 Insights from Similar Customers"):
        st.write("Here are profiles of customers similar to yours:")
        for idx in similar_customer_indices:
            st.write(f"- **Similar Customer {idx + 1}:**")
            st.write(f"  - Contract: **{X.iloc[idx]['Contract']}**")
            st.write(f"  - Monthly Charges: **${X.iloc[idx]['MonthlyCharges']:.2f}**")

    with st.expander("📈 Additional Retention Strategies"):
        st.write("To enhance retention, consider implementing these strategies:")
        st.write("- **Offer discounts on long-term contracts** to incentivize loyalty.")
        st.write("- **Promote online security or device protection services** to increase customer satisfaction.")
