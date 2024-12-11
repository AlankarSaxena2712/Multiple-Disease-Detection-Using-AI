import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn import preprocessing

import joblib

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# Load the saved model
model = load_model('diabetes_model.h5')


# Function to predict the outcome based on user input
def predict_outcome():
    print("Enter the values for the following features:")
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Collect input values from the user
    user_input = {}
    for feature in features:
        value = float(input(f"{feature}: "))
        user_input[feature] = [value]

    print("="*20)
    print("user_input:", user_input)
    print("="*20)
    
    user_input_df = pd.DataFrame(user_input)
    print("="*20)
    print("user_input_df:", user_input_df)
    print("="*20)
    
    # Scale the input using the pre-trained scaler
    user_input_scaled = scaler.transform(user_input_df)
    print("="*20)
    print("user_input_scaled:", user_input_scaled)
    print("="*20)
    
    # Predict the outcome
    prediction = model.predict(user_input_scaled)
    print("prediction:", prediction)
    outcome = "Diabetes" if prediction >= 0.5 else "No Diabetes"
    print(f"\nPrediction: {outcome} (Probability: {prediction[0][0]:.2f})")

# Call the function to predict outcome based on user input
predict_outcome()