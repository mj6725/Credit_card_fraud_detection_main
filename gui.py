# It creates a GUI for taking input from user or it can be automated.
# After collecting the input, it sends the data to the saved joblib model.
# The model analyses whether transaction is fraudulent or genuine using ML algorithms and returns the result.
from tkinter import *
import joblib
import pandas as pd
import numpy as np
from tkinter import messagebox

# Try to load the model
try:
    model = joblib.load('model_credit-card_fraud_detection.joblib')
    model_loaded = True
except:
    model_loaded = False
    print("Model file not found. Please run save_model.py first.")

window = Tk()
window.title("Credit Card Fraud Detection")
window.geometry("400x500")  # Increased size for better layout
canvas = Canvas(height=450, width=380, bg="#f0f0f0")
canvas.pack()

# Result label
result_label = Label(text="", font=("Arial", 12, "bold"))
result_label.place(x=100, y=375)

def predict():
    if not model_loaded:
        messagebox.showerror("Error", "Model not loaded. Run save_model.py first.")
        return
        
    try:
        # Get input values
        time = float(entry3.get())
        v1 = float(entry4.get())
        v2 = float(entry5.get())
        v14 = float(entry6.get())
        v28 = float(entry7.get())
        amount = float(entry8.get())
        
        # Create a dataframe with all needed features (31 columns)
        # Initialize with zeros for features we don't have UI inputs for
        features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        
        new_data = pd.DataFrame(0, index=[0], columns=features)
        
        # Set the values we have from the UI
        new_data['Time'] = time
        new_data['V1'] = v1
        new_data['V2'] = v2
        new_data['V14'] = v14
        new_data['V28'] = v28
        new_data['Amount'] = amount
        
        # Make prediction
        prediction = model.predict(new_data)[0]
        prediction_proba = model.predict_proba(new_data)[0]
        
        # Update result label
        if prediction == 1:
            result_text = f"FRAUD DETECTED! (Confidence: {prediction_proba[1]:.2%})"
            result_label.config(text=result_text, fg="red")
        else:
            result_text = f"Transaction is legitimate (Confidence: {prediction_proba[0]:.2%})"
            result_label.config(text=result_text, fg="green")
            
        print("Prediction:", "Fraud" if prediction == 1 else "Legitimate")
        print("Input data:", new_data)
            
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Title and instructions
title_label = Label(text="Credit Card Fraud Detection", font=("Arial", 14, "bold"))
title_label.place(x=70, y=10)

Label(text="Enter the following transaction data:", font=("Arial", 10)).place(x=70, y=40)

# Input fields
Label(text="Name:").place(x=20, y=80)
entry = Entry(font=("Arial", 12), width=15, borderwidth=3)
entry.place(x=180, y=80)

Label(text="Credit-Card No:").place(x=20, y=120)
entry2 = Entry(font=("Arial", 12), width=15, borderwidth=3)
entry2.place(x=180, y=120)

Label(text="Time:").place(x=20, y=160)
entry3 = Entry(font=("Arial", 12), width=15, borderwidth=3)
entry3.place(x=180, y=160)

Label(text="V1:").place(x=20, y=200)
entry4 = Entry(font=("Arial", 12), width=15, borderwidth=3)
entry4.place(x=180, y=200)

Label(text="V2:").place(x=20, y=240)
entry5 = Entry(font=("Arial", 12), width=15, borderwidth=3)
entry5.place(x=180, y=240)

Label(text="V14:").place(x=20, y=280)
entry6 = Entry(font=("Arial", 12), width=15, borderwidth=3)
entry6.place(x=180, y=280)

Label(text="V28:").place(x=20, y=320)
entry7 = Entry(font=("Arial", 12), width=15, borderwidth=3)
entry7.place(x=180, y=320)

Label(text="Amount:").place(x=20, y=360)
entry8 = Entry(font=("Arial", 12), width=15, borderwidth=3)
entry8.place(x=180, y=360)

# Add some example values for testing
entry3.insert(0, "0.0")    # Time
entry4.insert(0, "-1.35")  # V1
entry5.insert(0, "-0.07")  # V2
entry6.insert(0, "1.24")   # V14
entry7.insert(0, "-0.02")  # V28
entry8.insert(0, "149.62") # Amount

# Prediction button
Predict_button = Button(text="Predict", highlightthickness=0, bd=5, height=1, width=10, 
                       font=("Arial", 12, "bold"), command=predict, bg="#4CAF50", fg="white")
Predict_button.place(x=150, y=420)

# Show status message if model is not loaded
if not model_loaded:
    status = Label(text="⚠️ Model not loaded. Run save_model.py first.", fg="red")
    status.place(x=70, y=460)

window.mainloop()
