import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load model yang sudah disimpan
model = load_model('model/lstm.h5')

# Fungsi Preprocessing
def preprocess_name(name, max_len=20):
    name = name.lower()
    name_array = [max(0.0, ord(char) - 96.0) for char in name]
    name_array = name_array + [0.0] * (max_len - len(name_array))  # Pad to max_len
    return np.array(name_array).reshape(1, -1)

# Fungsi Prediksi
def predict_gender(name):
    name_array = preprocess_name(name)
    prediction = model.predict(name_array)
    prediction_prob = prediction[0][0]
    predicted_class = int(round(prediction_prob))
    
    if predicted_class == 1:
        gender = 'Laki-Laki'
        confidence_score = prediction_prob

    else:
        gender = 'Perempuan'
        confidence_score = 1 - prediction_prob

    return gender, confidence_score

# Contoh
name = "Putri"
predicted_gender, confidence_score = predict_gender(name)
print(f"Nama: {name}")
print(f"Prediksi Jenis Kelamin: {predicted_gender}")
print(f"Confidence Score: {confidence_score:.2f}")