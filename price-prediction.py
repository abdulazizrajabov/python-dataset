import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the preprocessor
preprocessor = joblib.load('preprocessor.pkl')

# Load the trained model
model = keras.models.load_model('used_car_price_model')

# Define function to load unique values
def load_unique_values(column_name):
    return pd.read_csv(f'options/{column_name}_unique.csv', header=None)[0].tolist()

# Define input fields
def user_input():
    st.sidebar.header('Car Specifications')

    manufacturer = st.sidebar.selectbox('Manufacturer', options=load_unique_values('manufacturer'))
    model = st.sidebar.selectbox('Model', options=load_unique_values('model'))
    year = st.sidebar.slider('Year', min_value=1990, max_value=2025)
    odometer = st.sidebar.slider('Odometer', min_value=0, max_value=300000)
    transmission = st.sidebar.selectbox('Transmission', options=load_unique_values('transmission'))

    data = {
        'manufacturer': manufacturer,
        'model': model,
        'year': year,
        'odometer': odometer,
        'transmission': transmission,
    }

    return pd.DataFrame(data, index=[0])

# Get user input
user_data = user_input()

# Preprocess the input
preprocessed_data = preprocessor.transform(user_data)

# Predict the car price
prediction = model.predict(preprocessed_data)

# Display the result
prediction_rounded = round(prediction[0][0]/100)*100
st.markdown(f"## The predicted price for the car with the given specifications is: **${prediction_rounded}**")
