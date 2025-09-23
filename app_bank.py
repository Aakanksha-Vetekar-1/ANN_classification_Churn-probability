import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open('label_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title('Customer churn Predictor')

geography = st.selectbox('Geography', categories_geo)
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.dataframe({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products], 
    'HasCrCard':[has_credit_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})
# ONE HOT ENCODE 'gEOGRAPHY
geo_encoded = onehot_encoder_geo.transform([[geography]]).to_array()
geo_encoded_df = pd.Dataframe(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis =1)

input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
churn_prob = prediction[0][0]

st.write(f'Churn probability: {churn_prob:.2f}')

if churn_prob > 0.5:
    print('Customer is likely to churn.')
else:
    print('Customer is unlikely to churn.')