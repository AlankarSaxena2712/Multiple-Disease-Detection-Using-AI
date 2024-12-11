import numpy as np

from keras.models import load_model

# Load the saved model
categorical_model = load_model('heart_disease_categorial_model.h5')
binary_model = load_model('heart_disease_binary_model.h5')

def get_user_input():
    """
    Collects and preprocesses user input for model prediction.
    """
    user_data = []
    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
        'ca', 'thal'
    ]
    
    print("Please provide the following inputs:")
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        user_data.append(value)

    # Convert to numpy array and reshape for prediction
    return np.array(user_data).reshape(1, -1)

# Predict using the categorical model
# user_input = get_user_input()


if __name__ == "__main__":
    user_input = get_user_input()
    print("user_input", user_input)
    categorical_prediction = categorical_model.predict(user_input)
    predicted_class = np.argmax(categorical_prediction, axis=1)
    print(f"Categorical Prediction (Class): {predicted_class[0]}")

    # Predict using the binary model
    binary_prediction = binary_model.predict(user_input)
    binary_outcome = "Heart Disease Detected" if binary_prediction[0][0] > 0.5 else "No Heart Disease Detected"
    print(f"Binary Prediction : {binary_outcome}, ({binary_prediction[0][0]})")