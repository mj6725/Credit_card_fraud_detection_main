
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load the model
@st.cache_resource
def load_model():
    try:
        return joblib.load('model_credit-card_fraud_detection.joblib')
    except:
        st.error("Model not found. Please run save_model.py first to train the model.")
        return None

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This application uses machine learning to detect fraudulent credit card transactions.
Enter the transaction details below to check if it's fraudulent or legitimate.
""")

# Load the model
model = load_model()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    # Input fields for important features
    time = st.number_input("Time (in seconds from first transaction)", min_value=0)
    amount = st.number_input("Transaction Amount ($)", min_value=0.0)
    
    # V1-V28 features with defaults
    st.subheader("Additional Features (V1-V28)")
    v_features = {}
    
    # Create 4 columns for V features
    v_cols = st.columns(4)
    for i in range(28):
        with v_cols[i % 4]:
            v_features[f'V{i+1}'] = st.number_input(
                f'V{i+1}',
                value=0.0,
                format="%.6f"
            )

    # Predict button
    if st.button("Check Transaction"):
        if model is not None:
            # Prepare input data
            input_data = pd.DataFrame({
                'Time': [time],
                'Amount': [amount],
                **{f'V{i+1}': [v_features[f'V{i+1}']] for i in range(28)}
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Display result
            with col2:
                st.subheader("Prediction Result")
                if prediction == 0:
                    st.success("✅ Legitimate Transaction")
                else:
                    st.error("🚨 Fraudulent Transaction")
                
                # Display probability
                st.write("Confidence Scores:")
                prob_df = pd.DataFrame({
                    'Transaction Type': ['Legitimate', 'Fraudulent'],
                    'Probability': [probability[0], probability[1]]
                })
                
                fig = px.bar(prob_df, x='Transaction Type', y='Probability',
                           color='Transaction Type', range_y=[0,1])
                st.plotly_chart(fig)

# Add information about the model
st.sidebar.title("About")
st.sidebar.info("""
This model was trained on a dataset of credit card transactions, 
with features that have been transformed using PCA for privacy reasons.
The model achieves high accuracy in detecting fraudulent transactions.

**Features:**
- Time: Seconds elapsed between transactions
- Amount: Transaction amount
- V1-V28: Transformed features (PCA)
""")

# Add sample transaction button
if st.sidebar.button("Load Sample Transaction"):
    # These are example values - you should replace with actual sample data
    st.session_state['sample_loaded'] = True
    sample_data = {
        'Time': 81622,
        'Amount': 146.73,
        'V1': -0.676435,
        'V2': -0.494024,
        'V3': 1.106210,
        'V4': 0.882661,
        'V5': -0.505835,
        # ... add remaining V features with sample values
    }
    st.experimental_rerun() 