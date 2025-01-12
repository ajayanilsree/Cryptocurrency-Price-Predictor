import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Function to load, preprocess, and return train-test split for a given dataset path
def preprocess_data(filepath):
    try:
        # Load dataset
        data = pd.read_csv(filepath)
        data = data.apply(pd.to_numeric, errors='coerce')  # Ensure correct data types

        # Fill missing values with column mean
        data.fillna(data.mean(), inplace=True)

        # Replace zeros in 'Volume' and 'Marketcap' with NaN and then fill with mean
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

    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")
        return None, None, None, None, None

# Function to build, train, save, and evaluate the KNN model with hyperparameter tuning
def train_and_evaluate(filepath, crypto_name):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(filepath)
    if X_train is None:
        return  # Exit if preprocessing failed

    try:
        # Initialize KNN regressor
        knn = KNeighborsRegressor()

        # Define the parameter grid for hyperparameter tuning
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13]}

        # Set up the GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')

        # Train the model with grid search
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_knn = grid_search.best_estimator_

        # Define the save directory and model filename
        save_directory = r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models'
        os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist

        # Save the trained model
        model_filename = os.path.join(save_directory, f"{crypto_name}_model.joblib")
        joblib.dump(best_knn, model_filename)

        # Save the scaler for consistent feature scaling
        scaler_filename = os.path.join(save_directory, f"{crypto_name}_scaler.joblib")
        joblib.dump(scaler, scaler_filename)

        # Evaluate on the test set
        test_score = best_knn.score(X_test, y_test)
        print(f"Test Score (R^2) for {crypto_name}: {test_score:.6f}")

        # Predict on the test set and calculate RMSE and MAE
        y_pred = best_knn.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Root Mean Squared Error (RMSE) for {crypto_name}: {rmse:.6f}")
        print(f"Mean Absolute Error (MAE) for {crypto_name}: {mae:.6f}")

        # Calculate R-squared
        r2 = r2_score(y_test, y_pred)
        print(f"R-squared (R^2) for {crypto_name}: {r2:.6f}")

        # Calculate accuracy based on ±5% margin
        margin = 0.05
        accurate_predictions = np.sum((y_pred >= (1 - margin) * y_test) & (y_pred <= (1 + margin) * y_test))
        accuracy = accurate_predictions / len(y_test) * 100
        print(f"Prediction Accuracy for {crypto_name}: {accuracy:.2f}%")

        # Plot observed vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(y_test)), y_test.values, label='Observed', color='blue', linewidth=2)
        plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
        plt.title(f'Observed vs Predicted Next Day Closing Price ({crypto_name})', fontsize=16)
        plt.xlabel('Sample Index', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"An error occurred with {crypto_name}: {e}")

# Paths to your datasets
crypto_datasets = {
    "Bitcoin": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Bitcoin.csv',
    "Ethereum": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Ethereum.csv',
    "Ripple": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_XRP.csv',
    "Tron": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Tron.csv',
    "Tether": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Tether.csv'
}

# Train and evaluate the KNN model for each cryptocurrency
for crypto_name, dataset_path in crypto_datasets.items():
    train_and_evaluate(dataset_path, crypto_name)

# Function to load a saved model and scaler
def load_model(crypto_name):
    try:
        # Load model and scaler
        model_filename = os.path.join(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models', f"{crypto_name}_model.joblib")
        scaler_filename = os.path.join(r'C:\Users\binge\OneDrive\Desktop\knn_crypt_models', f"{crypto_name}_scaler.joblib")
        
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        
        return model, scaler

    except Exception as e:
        print(f"An error occurred while loading the model for {crypto_name}: {e}")
        return None, None

# Function to make predictions
def make_prediction(model, data, scaler):
    # Scale the input data using the loaded scaler
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction[0]