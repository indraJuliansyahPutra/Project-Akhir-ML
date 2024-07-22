import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# Load the saved models
lstm_model = load_model('model/lstm.h5')
knn_model = joblib.load('model/knn.pkl')
rf_model = joblib.load('model/rf.pkl')
nb_model = joblib.load('model/nb.pkl')
dt_model = joblib.load('model/dt.pkl')

# Preprocess function
def preprocess_name(name, max_len=20):
    name = name.lower()
    name_array = list(name) + [' '] * (max_len - len(name))
    name_array = [max(0.0, ord(char)-96.0) for char in name_array]
    return np.array(name_array).reshape(1, -1)

# Predict function for LSTM
def predict_gender_lstm(name):
    name_array = preprocess_name(name)
    prediction = lstm_model.predict(name_array)
    predicted_class = int(round(prediction[0][0]))
    confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
    return 'Laki-Laki' if predicted_class == 1 else 'Perempuan', confidence

# Predict function for KNN, RF, and NB
def predict_gender_model(name, model):
    name_array = preprocess_name(name)
    prediction = model.predict_proba(name_array)
    predicted_class = model.predict(name_array)[0]
    confidence = max(prediction[0])
    return 'Laki-Laki' if predicted_class == 1 else 'Perempuan', confidence

# Function to display images in Graph page
def display_images(image_paths):
    # Membuat dua kolom
    col1, col2 = st.columns(2)
    
    # Menampilkan gambar di kolom pertama
    with col1:
        # Menampilkan gambar dari setengah pertama daftar
        for path in image_paths[:len(image_paths)//2]:
            image = Image.open(path)
            st.image(image, caption=path)
    
    # Menampilkan gambar di kolom kedua
    with col2:
        # Menampilkan gambar dari setengah kedua daftar
        for path in image_paths[len(image_paths)//2:]:
            image = Image.open(path)
            st.image(image, caption=path)

# Streamlit UI
st.title("Name Gender Predictor")

# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Home", "Prediksi", "Grafik"])

if page == "Home":
    st.header("Selamat Datang di Aplikasi Name Gender Predictor")
    st.write("""
    Aplikasi ini menggunakan model machine learning untuk memprediksi jenis kelamin seseorang berdasarkan nama.
    Model yang digunakan adalah LSTM (Long Short-Term Memory), KNN (K-Nearest Neighbors), Random Forest, dan Naive Bayes.
    Anda dapat memasukkan nama dan memilih model untuk mendapatkan prediksi jenis kelamin beserta nilai confidence-nya.
    """)
elif page == "Prediksi":
    st.header("Halaman Prediksi")

    # Input name from user
    name = st.text_input("Masukkan Nama:", "")

    # Model selection
    model_option = st.selectbox(
        "Pilih Model:",
        ("LSTM", "KNN", "Random Forest", "Naive Bayes", "Decision Tree", "Semuanya")
    )

    # Predict button
    if st.button("Prediksi"):
        if name:
            st.write(f'Nama: {name}')
            if model_option == "LSTM" or model_option == "Semuanya":
                predicted_gender_lstm, confidence_lstm = predict_gender_lstm(name)
                st.write(f"Prediksi Jenis Kelamin (LSTM): {predicted_gender_lstm} dengan confidence {confidence_lstm:.2f}")

            if model_option == "KNN" or model_option == "Semuanya":
                predicted_gender_knn, confidence_knn = predict_gender_model(name, knn_model)
                st.write(f"Prediksi Jenis Kelamin (KNN): {predicted_gender_knn} dengan confidence {confidence_knn:.2f}")

            if model_option == "Random Forest" or model_option == "Semuanya":
                predicted_gender_rf, confidence_rf = predict_gender_model(name, rf_model)
                st.write(f"Prediksi Jenis Kelamin (Random Forest): {predicted_gender_rf} dengan confidence {confidence_rf:.2f}")

            if model_option == "Naive Bayes" or model_option == "Semuanya":
                predicted_gender_nb, confidence_nb = predict_gender_model(name, nb_model)
                st.write(f"Prediksi Jenis Kelamin (Naive Bayes): {predicted_gender_nb} dengan confidence {confidence_nb:.2f}")

            if model_option == "Decision Tree" or model_option == "Semuanya":
                predicted_gender_dt, confidence_dt = predict_gender_model(name, dt_model)
                st.write(f"Prediksi Jenis Kelamin (Decision Tree): {predicted_gender_dt} dengan confidence {confidence_dt:.2f}")


elif page == "Grafik":
    st.header("Grafik Hasil Pelatihan")
    image_paths = ["grafik/acc.png", "grafik/loss.png", "grafik/confusion_matrix.png", "grafik/confusion_matrix_knn.png", "grafik/confusion_matrix_rf.png", "grafik/confusion_matrix_nb.png", "grafik/confusion_matrix_dt.png"]
    display_images(image_paths)
