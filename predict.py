import pandas as pd
import numpy as np
import joblib

# Load model yang sudah disimpan
model_file = 'model/nb.pkl'
best_model_knn = joblib.load(model_file)

# Fungsi Preprocessing
def preprocess_name(name, max_len=20):
    name = name.lower()
    name_array = list(name) + [' '] * (max_len - len(name))
    name_array = [max(0.0, ord(char)-96.0) for char in name_array]
    return np.array(name_array).reshape(1, -1)

# Fungsi Prediksi
def predict_gender(name):
    name_array = preprocess_name(name)
    prediction_proba = best_model_knn.predict_proba(name_array)[0]
    prediction = best_model_knn.predict(name_array)[0]
    
    if prediction == 1:
        gender = 'Laki-Laki'
        confidence_score = prediction_proba[1]

    else:
        gender = 'Perempuan'
        confidence_score = prediction_proba[0]

    return gender, confidence_score

# Contoh
name = "Putri" 
predicted_gender, confidence_score = predict_gender(name)
print(f"Nama: {name}")
print(f"Prediksi Jenis Kelamin: {predicted_gender}")
print(f"Confidence Score: {confidence_score:.2f}")