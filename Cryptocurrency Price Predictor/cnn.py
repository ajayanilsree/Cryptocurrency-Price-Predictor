import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Function to load, preprocess, and return train-test split for a given dataset path
def preprocess_data(filepath):
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
    
    # Reshape X for Conv1D: (samples, features, 1) for PyTorch
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    # Split into training and testing sets (70:30)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Convert to PyTorch tensors
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)
    
    return X_train, X_test, y_train, y_test

# CNN model in PyTorch
class CryptoCNN(nn.Module):
    def __init__(self):
        super(CryptoCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)  # Adjust input size to match the output from Conv layers
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# Function to train and evaluate the model
def train_and_evaluate(filepath, model_path, crypto_name):
    X_train, X_test, y_train, y_test = preprocess_data(filepath)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model, loss function, and optimizer
    model = CryptoCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_preds = []
        y_tests = []
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch).squeeze()
            y_preds.append(y_pred.view(-1))
            y_tests.append(y_batch)
    
    # Convert predictions to a single tensor
    y_pred = torch.cat(y_preds).numpy()
    y_test = torch.cat(y_tests).numpy()
    
    # Calculate RMSE
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE) for {crypto_name}: {rmse}")
    
    # Plot observed vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Observed (Actual)', color='blue')
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

# Paths to save the models
model_paths = {
    "Bitcoin": r'C:\Users\binge\OneDrive\Desktop\cnn_crypt_models\bitcoin_cnn.pth',
    "Ethereum": r'C:\Users\binge\OneDrive\Desktop\cnn_crypt_models\ethereum_cnn.pth',
    "Ripple": r'C:\Users\binge\OneDrive\Desktop\cnn_crypt_models\ripple_cnn.pth',
    "Tron": r'C:\Users\binge\OneDrive\Desktop\cnn_crypt_models\tron_cnn.pth',
    "Tether": r'C:\Users\binge\OneDrive\Desktop\cnn_crypt_models\tether_cnn.pth'
}

# Train and evaluate the CNN model for each cryptocurrency
for crypto_name, dataset_path in crypto_datasets.items():
    train_and_evaluate(dataset_path, model_paths[crypto_name], crypto_name)
