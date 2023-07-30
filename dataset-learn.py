import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import tensorflow as tf
from tensorflow import keras

# Function to convert scipy sparse matrix to tensorflow SparseTensor
def sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.SparseTensor(indices, coo.data, coo.shape)

# Load data
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('vehicles-new.csv')  # Replace with your csv path
    return df

# Preprocess data
def preprocess_data(df):
    # Categorical features to be one-hot encoded
    categorical_features = ['manufacturer', 'model',
                            'transmission']
    # Numeric features to be scaled
    numeric_features = ['year', 'odometer']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse=True, handle_unknown='ignore'), categorical_features)],
            remainder='passthrough')

    features = df[categorical_features + numeric_features]
    labels = df['price']

    # Split the data into training and testing sets
    X, X_test, y, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Normalize the features
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    # Save preprocessor
    joblib.dump(preprocessor, 'preprocessor.pkl')

    return X_train, X_val, X_test, y_train, y_val, y_test

df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

# Convert sparse matrices to sparse tensors
X_train_tensor = sparse_matrix_to_sparse_tensor(X_train)
X_val_tensor = sparse_matrix_to_sparse_tensor(X_val)

# Create tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, y_train.values)).batch(512)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_tensor, y_val.values)).batch(512)

# Model architecture
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    return model

# Train model
if st.button('Train Model'):
    model = build_model()
    model.fit(train_dataset, epochs=10, validation_data=val_dataset, verbose=0)
    # Save model
    model.save('used_car_price_model')
    st.write("Model Trained and Saved Successfully!")
