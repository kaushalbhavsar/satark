# LSTM-Based Insider Threat Anomaly Detection

**Author:** Dr. Kaushal Bhavsar
**License:** MIT
## Overview
This project demonstrates how to use an LSTM (Long Short-Term Memory) neural network for unsupervised anomaly detection in time-series security logs—such as USB activity, file reads/writes, or other endpoint behaviors. The goal is to identify unusual user activity that may signal insider threats or data exfiltration, using only behavioral data.

## Features

- Unsupervised anomaly detection—no labels required
- Works on any sequential security log data (USB, files, logins, etc.)
- Clean, modular Python code (Keras/TensorFlow)
- Visualizes anomaly scores and flags outliers
- Saves detected anomalies for further analysis
## Getting Started

### 1. **Clone the repository**

`git clone https://github.com/kbhavsar/satark.git`
`cd lstm-insider-threat-detection`

### 2. Install dependencies

`pip install -r requirements.txt`

**Dependencies:**
`numpy`
`pandas`
`scikit-learn`
`matplotlib`
`tensorflow`
### 3. Prepare your data

Prepare a CSV file (e.g., security_logs.csv) with at least:

A timestamp column (e.g., timestamp)
One or more feature columns (e.g., usb_events, file_reads, file_writes)
Edit the script to update:
DATA_PATH
TIMESTAMP_COL
FEATURES (list of feature columns to use)
### 4. Run the script

`python satark_usb.py`

The script will train an LSTM model, compute anomaly scores, visualize the results, and export detected anomalies to anomalies_detected.csv.

## Example Visualization

Sample output: higher points and red markers indicate detected anomalies.

## How It Works

1. **Preprocessing**: Scales features and creates rolling sequences for LSTM.
2. **Model**: Trains an LSTM to predict next-step behavior based on previous events.
3.  **Scoring**: Calculates reconstruction error (MSE) as the anomaly score.
4. **Thresholding**: Flags events as anomalies if their score exceeds a set percentile (default: 99th).
5. **Output**: Saves anomalies and shows a plot for review.

## Customization

Change the sequence length, features, and LSTM architecture as needed.
Plug in any time-series log data—just update FEATURES and the CSV input.

Threshold and visualization can be tuned for your use-case.

## License

MIT

## Author

Dr. Kaushal Bhavsar - https://bhavsar.ai


## Contributions

If you improve this pipeline (add transformer models, new features, or integrations), feel free to open a pull request. 

  

**Disclaimer**

This project is for educational and research use. It is not a replacement for commercial-grade security monitoring.
