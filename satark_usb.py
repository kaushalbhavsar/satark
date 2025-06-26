"""
Satark-v1: LSTM-based Anomaly Detection for Insider Threats
Author: Dr. Kaushal Bhavsar
License: MIT

This script demonstrates how to build and use an LSTM neural network to detect
anomalies in time-series security data (e.g., USB activity, file operations).

Usage:
- Prepare a CSV file of log data with time-based features.
- Modify 'FEATURES' as needed.
- Run the script: python lstm_anomaly_detection.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION ---

DATA_PATH = "security_logs.csv"    # Path to your CSV file
TIMESTAMP_COL = "timestamp"        # Name of timestamp column
FEATURES = ["usb_events", "file_reads", "file_writes"]  # Change as needed
SEQUENCE_LENGTH = 20               # How many time steps per sequence
EPOCHS = 25
BATCH_SIZE = 32

# --- DATA PREP ---

# Load and sort data
df = pd.read_csv(DATA_PATH, parse_dates=[TIMESTAMP_COL])
df = df.sort_values(TIMESTAMP_COL)

# Fill missing values (simple)
df[FEATURES] = df[FEATURES].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:(i+seq_length)]
        xs.append(x)
    return np.array(xs)

X_seq = create_sequences(X_scaled, SEQUENCE_LENGTH)

# --- LSTM MODEL ---

model = Sequential([
    LSTM(64, input_shape=(SEQUENCE_LENGTH, len(FEATURES)), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(len(FEATURES))
])
model.compile(optimizer='adam', loss='mse')

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

# Train (unsupervised: target = input)
history = model.fit(
    X_seq, X_seq[:, -1, :],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=[early_stopping]
)

# --- ANOMALY DETECTION ---

# Predict last value in each sequence
X_pred = model.predict(X_seq)
# Calculate MSE (error between predicted and actual)
mse = np.mean(np.power(X_pred - X_seq[:, -1, :], 2), axis=1)

# Determine anomaly threshold (e.g., 99th percentile)
threshold = np.percentile(mse, 99)
anomalies = mse > threshold

# Add results to DataFrame
anomaly_col = np.full(len(df), False)
anomaly_col[SEQUENCE_LENGTH:] = anomalies
df["anomaly"] = anomaly_col

# --- VISUALIZE ---

plt.figure(figsize=(15,4))
plt.plot(df[TIMESTAMP_COL], mse, label="Anomaly Score (MSE)")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
plt.scatter(df[TIMESTAMP_COL][df["anomaly"]], mse[anomalies], color='red', marker='x', label="Anomalies")
plt.title("LSTM Anomaly Detection (Insider Threat)")
plt.xlabel("Time")
plt.ylabel("Anomaly Score")
plt.legend()
plt.tight_layout()
plt.show()

# --- EXPORT ANOMALIES ---

df[df["anomaly"]].to_csv("anomalies_detected.csv", index=False)

print(f"Anomaly detection complete. {df['anomaly'].sum()} anomalies found and saved to anomalies_detected.csv")

# --- END OF SCRIPT ---
