import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib  # Used to load the KNN models

# Load KNN models and scalers for each cryptocurrency
models = {
    'Bitcoin': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Bitcoin_model.joblib'),
    'Ethereum': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Ethereum_model.joblib'),
    'Ripple': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Ripple_model.joblib'),
    'Tron': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Tron_model.joblib'),
    'Tether': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Tether_model.joblib')
}

# Load scalers for each cryptocurrency
scalers = {
    'Bitcoin': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Bitcoin_scaler.joblib'),
    'Ethereum': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Ethereum_scaler.joblib'),
    'Ripple': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Ripple_scaler.joblib'),
    'Tron': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Tron_scaler.joblib'),
    'Tether': joblib.load(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models\Tether_scaler.joblib')
}

# Function to make predictions using the selected KNN model
def make_prediction(model, scaler, data):
    # Scale the input data
    data_scaled = scaler.transform(data)
    
    # Make the prediction
    prediction = model.predict(data_scaled)
    return prediction[0]

# Streamlit app UI
st.title('CoinVision: Crypto’s Next Move')

# Select which cryptocurrency to predict
coin = st.selectbox('Select Cryptocurrency', ['Bitcoin', 'Ethereum', 'Ripple', 'Tron', 'Tether'])

# Input fields for user to enter data
st.subheader(f'Enter the details for {coin}')
high = st.number_input('High', min_value=0.0, step=0.0001, format="%.4f")
low = st.number_input('Low', min_value=0.0, step=0.0001, format="%.4f")
open_price = st.number_input('Open', min_value=0.0, step=0.0001, format="%.4f")
close = st.number_input('Close', min_value=0.0, step=0.0001, format="%.4f")
volume = st.number_input('Volume', min_value=0.0, step=0.0001, format="%.4f")
marketcap = st.number_input('Marketcap', min_value=0.0, step=0.0001, format="%.4f")

# Create a DataFrame with input data
data = np.array([[high, low, open_price, close, volume, marketcap]])
data_df = pd.DataFrame(data, columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap'])

# Prediction logic
if st.button('Predict Next Day Close Price'):
    if high < 0 or low < 0 or open_price < 0 or close < 0 or volume < 0 or marketcap < 0:
        st.error("Values must be positive.")
    else:
        # Load the model and scaler based on the selected cryptocurrency
        model = models[coin]
        scaler = scalers[coin]
        
        # Make the prediction
        prediction = make_prediction(model, scaler, data_df)
        
        # Display the result
        st.subheader(f'Predicted Next Day Close Price for {coin}: ${prediction:.4f}')
