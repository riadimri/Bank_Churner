import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load your pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the StandardScaler
scaler = StandardScaler()

# Define a function to make predictions
def predict_churn(data):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction

# Streamlit app
st.title('Bank Customer Churn Prediction')

# Accept user inputs
customer_id = st.text_input('Customer ID')
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
country = st.selectbox('Country', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, value=10000.0)
products_number = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
credit_card = st.selectbox('Has Credit Card?', [0, 1])
active_member = st.selectbox('Active Member?', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

# Create a dataframe from user inputs
input_data = pd.DataFrame({
    'customer_id': [customer_id],
    'credit_score': [credit_score],
    'country': [country],
    'gender': [gender],
    'age': [age],
    'tenure': [tenure],
    'balance': [balance],
    'products_number': [products_number],
    'credit_card': [credit_card],
    'active_member': [active_member],
    'estimated_salary': [estimated_salary]
})

# Encoding categorical variables
input_data['country'] = input_data['country'].map({'France': 0, 'Germany': 1, 'Spain': 2})
input_data['gender'] = input_data['gender'].map({'Male': 0, 'Female': 1})

# Drop customer_id column
input_data = input_data.drop('customer_id', axis=1)

# Display the input dataframe
st.write('Input Data')
st.write(input_data)

# Make prediction
if st.button('Predict'):
    result = predict_churn(input_data)
    st.write('Prediction (1 means the customer will churn, 0 means they will not):')
    st.write(result[0])
