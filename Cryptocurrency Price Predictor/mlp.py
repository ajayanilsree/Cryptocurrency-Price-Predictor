import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the neural network model in PyTorch
class CryptoModel(nn.Module):
    def __init__(self):
        super(CryptoModel, self).__init__()
        self.layer1 = nn.Linear(6, 6)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(6, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.dropout1(x)
        x = torch.tanh(self.layer2(x))
        x = self.dropout2(x)
        x = torch.tanh(self.layer3(x))
        x = self.dropout3(x)
        x = self.output(x)
        return x

def train_and_evaluate_crypto_model(filepath, coin_name):
    # Load dataset
    data = pd.read_csv(filepath)
    data = data.apply(pd.to_numeric, errors='coerce')

    # Handle NaN values
    if data.isnull().any().any():
        print(f"{coin_name}: Dataset contains NaN values. Filling missing values.")
        data = data.fillna(data.mean())
    else:
        print(f"{coin_name}: No missing values detected.")

    # Replace zeros in critical columns and handle missing values
    data[['Volume', 'Marketcap']] = data[['Volume', 'Marketcap']].replace(0, np.nan)
    data = data.fillna(data.mean())

    # Set independent variables and dependent variable
    X = data[['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']]
    data['Predictions'] = data['Close'].shift(-1)
    y = data['Predictions'].dropna()
    X = X.iloc[:-1]

    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)

    # Initialize the model, loss function, and optimizer
    model = CryptoModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    epochs = 50
    batch_size = 32
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0.0
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), rf"C:\Users\binge\OneDrive\Desktop\mlp_crypt_models/{coin_name.lower()}.pth")

    # Evaluation
       # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).flatten()  # Flatten the predictions to match y_test's shape
        mse = criterion(y_pred, y_test.flatten()).item()  # Flatten y_test to make sure it matches y_pred
        rmse = np.sqrt(mse)

    print(f"{coin_name} Test Loss (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot observed vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.numpy(), label='Observed (Actual)', color='blue')
    plt.plot(y_pred.numpy(), label='Predicted', color='red', linestyle='--')
    plt.title(f'Observed vs Predicted Next Day Closing Price ({coin_name})')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Paths and coin names
crypto_datasets = {
    "Bitcoin": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Bitcoin.csv',
    "Ethereum": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Ethereum.csv',
    "Ripple": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_XRP.csv',
    "Tron": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Tron.csv',
    "Tether": r'C:\Users\binge\OneDrive\Desktop\crypt_dataset\coin_Tether.csv'
}

# Run the model for each cryptocurrency
for coin, path in crypto_datasets.items():
    train_and_evaluate_crypto_model(path, coin)
