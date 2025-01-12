import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR  # Import SVR for Support Vector Regression
import matplotlib.pyplot as plt
import joblib  # Import joblib for saving models
import os  # Import os for path management

# Function to load, preprocess, and return train-test split for a given dataset path
def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    # Convert to numeric types and handle missing data
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(data.mean(), inplace=True)
    
    # Replace zeros in 'Volume' and 'Marketcap' with NaN, then fill with mean
    data[['Volume', 'Marketcap']] = data[['Volume', 'Marketcap']].replace(0, np.nan)
    data.fillna(data.mean(), inplace=True)
    
    # Define independent variables (X) and dependent variable (y)
    X = data[['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']]
    data['Predictions'] = data['Close'].shift(-1)
    y = data['Predictions'].dropna()
    X = X.iloc[:-1]  # Align X with shifted y
    
    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets (70:30)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

# Function to build, train, save, and evaluate the SVM model
def train_and_evaluate(filepath, crypto_name):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(filepath)
    
    # Initialize SVM regressor
    svm = SVR(kernel='rbf')  # You can tune the kernel and other parameters
    
    # Train the SVM model
    svm.fit(X_train, y_train)
    
    # Define the save directory and model filename
    save_directory = r'C:\Users\binge\OneDrive\Desktop\svm_crypt_models'  # Keep the same directory for consistency
    os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
    model_filename = os.path.join(save_directory, f"{crypto_name}_svm_model.joblib")
    
    # Save the trained model
    joblib.dump(svm, model_filename)
    
    # Save the scaler for consistent future predictions
    scaler_filename = os.path.join(save_directory, f"{crypto_name}_scaler.joblib")
    joblib.dump(scaler, scaler_filename)
    
    # Evaluate on the test data
    test_score = svm.score(X_test, y_test)
    print(f"\n--- Results for {crypto_name} ---")
    print(f"Test Score (R^2): {test_score:.6f}")
    
    # Predict on the test set
    y_pred = svm.predict(X_test)
    
    # Calculate RMSE
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    
    # Plot observed vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Observed (Actual)', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(f'Observed vs Predicted Next Day Closing Price ({crypto_name})')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Paths to your datasets
crypto_datasets = {
    "Bitcoin": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Bitcoin.csv',
    "Ethereum": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Ethereum.csv',
    "Ripple": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_XRP.csv',
    "Tron": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Tron.csv',
    "Tether": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Tether.csv'
}

# Train and evaluate the SVM model for each cryptocurrency
for crypto_name, dataset_path in crypto_datasets.items():
    train_and_evaluate(dataset_path, crypto_name)

# Function to load a saved model and scaler
def load_model(crypto_name):
    model_filename = os.path.join(r'C:\Users\binge\OneDrive\Desktop\svm_crypt_models', f"{crypto_name}_svm_model.joblib")
    scaler_filename = os.path.join(r'C:\Users\binge\OneDrive\Desktop\svm_crypt_models', f"{crypto_name}_scaler.joblib")
    
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    return model, scaler
