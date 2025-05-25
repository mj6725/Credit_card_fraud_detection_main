# Credit Card Fraud Detection

## Project Overview

This project implements a machine learning solution to identify and prevent fraudulent credit card transactions. The system uses a Random Forest Classifier trained on transaction data to monitor and flag suspicious activities in real-time.

### Key Features

- **Machine Learning Detection**: Uses Random Forest algorithm to detect fraud with 99.95% accuracy
- **Real-time Analysis**: Analyzes transaction features to identify suspicious patterns
- **User-friendly Interface**: Simple GUI for testing transaction fraud detection

## Dataset

The model is trained on a dataset of credit card transactions containing:
- 284,807 transactions with 31 features
- 492 fraudulent transactions (0.17%)
- 284,315 legitimate transactions (99.83%)

Key transaction features include:
- Time: Seconds elapsed between transactions
- V1-V28: PCA-transformed features (anonymized for security)
- Amount: Transaction amount
- Class: 1 for fraud, 0 for legitimate transaction

## Installation and Usage

### Prerequisites
- Python 3.6+
- Required libraries: scikit-learn, pandas, numpy, joblib, tkinter

### Setup
1. Clone or download this repository
2. Install required dependencies:
   ```
   pip install scikit-learn pandas numpy joblib
   ```
3. Ensure the dataset 'creditcard.csv' is in the project root directory

### Running the Application
1. First, train and save the model:
   ```
   python save_model.py
   ```
2. Launch the GUI application:
   ```
   python gui.py
   ```
3. Enter transaction details in the GUI and click "Predict" to check if it's fraudulent

## Implementation Details

- `Credit_Card_Fraud_Detection.ipynb`: Jupyter notebook with data exploration and model development
- `save_model.py`: Script to train and save the Random Forest model
- `gui.py`: Tkinter GUI for fraud detection testing
- `model_credit-card_fraud_detection.joblib`: Saved model file (generated after running save_model.py)

## Performance

- Random Forest Classifier: 99.95% accuracy
- Decision Tree Regressor: 99.92% accuracy

## Future Improvements

- Implement real-time transaction monitoring
- Add more visualization features for better interpretability
- Enhance GUI with more detailed analysis of suspicious factors
- Deploy as a web service for remote access
