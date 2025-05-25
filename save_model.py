# Code for saving the machine learning model which will then be connected with the GUI
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Prepare the data
X = data.drop(['Class'], axis=1)
y = data['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)

# Train the model
print("Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the model
print("Saving model to 'model_credit-card_fraud_detection.joblib'")
joblib.dump(rf, 'model_credit-card_fraud_detection.joblib')

# Test loading the model
print("Testing model loading...")
loaded_model = joblib.load('model_credit-card_fraud_detection.joblib')

# Confirm model works with a simple prediction test
print("Testing prediction on a sample from test data...")
sample = X_test.iloc[:1]
prediction = loaded_model.predict(sample)
print(f"Prediction for test sample: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")
print("Model saved successfully!")
