import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib  # Used to load the KNN models

# Load KNN models for each cryptocurrency
models = {
    'Bitcoin': joblib.load(r'C:\Users\HP\Desktop\knn_crypt_models\Bitcoin_model.joblib'),
    'Ethereum': joblib.load(r'C:\Users\HP\Desktop\knn_crypt_models\Ethereum_model.joblib'),
    'Ripple': joblib.load(r'C:\Users\HP\Desktop\knn_crypt_models\Ripple_model.joblib'),
    'Tron': joblib.load(r'C:\Users\HP\Desktop\knn_crypt_models\Tron_model.joblib'),
    'Tether': joblib.load(r'C:\Users\HP\Desktop\knn_crypt_models\Tether_model.joblib')
}

# Function to make predictions using the selected KNN model
def make_prediction(model, data):
    # Define MinMaxScaler to scale the input data
    scaler = MinMaxScaler()
    
    # Fit the scaler on the training data used during model training
    min_values = [0, 0, 0, 0, 0, 0]  # Min values of [High, Low, Open, Close, Volume, Marketcap]
    max_values = [10000, 10000, 10000, 10000, 10000000, 100000000]  # Max values of [High, Low, Open, Close, Volume, Marketcap]
    
    # Fit the scaler based on these values
    scaler.fit([min_values, max_values])
    
    # Scale the input data
    data_scaled = scaler.transform(data)
    
    # Make the prediction
    prediction = model.predict(data_scaled)
    return prediction[0]

# Streamlit app UI
st.set_page_config(page_title='CoinVision', layout='centered')  # Set page title and layout
st.title('🌟 CoinVision: Crypto’s Next Move 🌟')

# Add images
st.image("https://example.com/crypto_image.jpg", width=700)  # Replace with actual image URL

# Select which cryptocurrency to predict
st.subheader('Select Cryptocurrency')
coin = st.selectbox('', ['Bitcoin', 'Ethereum', 'Ripple', 'Tron', 'Tether'], 
                      format_func=lambda x: x, key="crypto_select")

# Input fields for user to enter data
st.subheader(f'Enter the details for {coin}')
high = st.number_input('High', min_value=0.0, step=0.01)
low = st.number_input('Low', min_value=0.0, step=0.01)
open_price = st.number_input('Open', min_value=0.0, step=0.01)
close = st.number_input('Close', min_value=0.0, step=0.01)
volume = st.number_input('Volume', min_value=0.0, step=0.01)
marketcap = st.number_input('Marketcap', min_value=0.0, step=0.01)

# Create a DataFrame with input data
data = np.array([[high, low, open_price, close, volume, marketcap]])
data_df = pd.DataFrame(data, columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap'])

# Prediction logic
if st.button('🔮 Predict Next Day Close Price'):
    if high < 0 or low < 0 or open_price < 0 or close < 0 or volume < 0 or marketcap < 0:
        st.error("⚠️ Values must be positive.")
    else:
        # Load the model based on the selected cryptocurrency
        model = models[coin]
        
        # Make the prediction
        prediction = make_prediction(model, data_df)
        
        # Display the result
        st.success(f'💰 Predicted Next Day Close Price for {coin}: **${prediction:.2f}**')

# Styling for better UI experience
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f2f5;  /* Light gray background */
            color: #333;  /* Dark text for better readability */
            font-family: 'Arial', sans-serif;
        }
        h1 {
            font-size: 2.5rem;
            font-weight: bold;
            color: #4CAF50;  /* Green color for title */
            text-align: center;
        }
        h2, h3, h4 {
            color: #333;  /* Dark text for subheaders */
        }
        .stNumberInput label {
            color: black;  /* Black text for input labels */
        }
        .stSelectbox label {
            color: black;  /* Black text for select label */
        }
        .stButton {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 1.2rem;
        }
        .stButton:hover {
            background-color: #45a049;
        }
        input[type="number"] {
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 10px;
            font-size: 1rem;
        }
        .stError, .stSuccess {
            font-size: 1.2rem;
            text-align: center;
        }
        .stSelectbox {
            font-size: 1.2rem;
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        /* Reduce the width of the main content */
        .stContainer {
            max-width: 700px;  /* Adjust the width */
            margin: auto;  /* Center the container */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add images of cryptocurrencies
st.image("https://example.com/bitcoin.png", caption="Bitcoin", width=100)  # Replace with actual image URL
st.image("https://example.com/ethereum.png", caption="Ethereum", width=100)  # Replace with actual image URL
st.image("https://example.com/ripple.png", caption="Ripple", width=100)  # Replace with actual image URL
st.image("https://example.com/tron.png", caption="Tron", width=100)  # Replace with actual image URL
st.image("https://example.com/tether.png", caption="Tether", width=100)  # Replace with actual image URL
