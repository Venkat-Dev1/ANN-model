import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st
import tensorflow as tf
import pickle as pkl

# Load the trained model and encoders
model = tf.keras.models.load_model('churn_model.h5', compile=False)
with open('scaler.pkl', 'rb') as f:
    scaler = pkl.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pkl.load(f)
with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder = pkl.load(f)

# Streamlit app
st.title('Customer Churn Prediction')

# User input
geo = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', min_value=18, max_value=100, value=30)
tenure = st.slider('Tenure (Years)', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, value=10000.0)
Credit_Score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=10, value=1)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

input_data = pd.DataFrame({
    'CreditScore': [Credit_Score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary],
})

geo_encoded = one_hot_encoder.transform(pd.DataFrame({'Geography': [geo]})).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geo_df], axis=1)


# Ensure exact feature order expected by scaler/model pipeline.
input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]
    st.write(f'Churn Probability: {churn_probability:.2f}')
    if churn_probability > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is unlikely to churn.')
