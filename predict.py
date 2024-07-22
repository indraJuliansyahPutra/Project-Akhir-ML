import pandas as pd
import numpy as np
import joblib

# Load the saved model
model_file = 'model/nb.pkl'
best_model_knn = joblib.load(model_file)

# Function to preprocess a single name
def preprocess_name(name, max_len=20):
    name = name.lower()
    name_array = list(name) + [' '] * (max_len - len(name))
    name_array = [max(0.0, ord(char)-96.0) for char in name_array]
    return np.array(name_array).reshape(1, -1)

# Predict function
def predict_gender(name):
    name_array = preprocess_name(name)
    prediction = best_model_knn.predict(name_array)
    return 'Laki-Laki' if prediction[0] == 1 else 'Perempuan'

# Example prediction
name = "Putri"  # Ganti dengan nama lain untuk prediksi
predicted_gender = predict_gender(name)
print(f"Nama: {name}")
print(f"Prediksi Jenis Kelamin: {predicted_gender}")