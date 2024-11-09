# Customer Churn Prediction and Recommendation System

This application is a **Customer Churn Prediction and Recommendation System** built using **Streamlit** and **CatBoost**. It leverages customer data to predict the likelihood of churn and provides personalized recommendations to improve retention. The recommendations are generated based on similar customer profiles and key factors associated with churn, helping businesses make data-driven decisions to retain customers.

## Overview

The application allows users to:
- Predict if a customer is likely to churn based on inputs like gender, tenure, monthly charges, and more.
- Generate a personalized set of recommendations aimed at reducing churn probability.
- Display similar customer profiles to offer additional insights for retention strategies.

The model used is a **CatBoost Classifier**, trained on customer churn data and saved as a pickle file (`churn_prediction_model.pkl`). This application also uses **cosine similarity** to match the input customer profile with similar customers in the dataset, offering targeted recommendations based on their traits.

## Prerequisites

Make sure you have the following installed:
- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy
- joblib
- catboost

## Getting Started

1. **Install dependencies** :
   ```bash
   pip install -r requirements.txt
2.**Run the app** :
  ```bash
    streamlit run app.py
