import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model('gender_prediction.h5')

# Function to preprocess a single name
def preprocess_name(name, max_len=20):
    name = name.lower()
    name_array = [max(0.0, ord(char) - 96.0) for char in name]
    name_array = name_array + [0.0] * (max_len - len(name_array))  # Pad to max_len
    return np.array(name_array).reshape(1, -1)

# Predict function
def predict_gender(name):
    name_array = preprocess_name(name)
    prediction = model.predict(name_array)
    predicted_class = int(round(prediction[0][0]))
    return 'Laki-Laki' if prediction == 1 else 'Perempuan'

# Example prediction
name = "Zee"  # Ganti dengan nama lain untuk prediksi
predicted_gender = predict_gender(name)
print(f"Nama: {name}")
print(f"Prediksi Jenis Kelamin: {predicted_gender}")