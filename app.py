import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Judul halaman
st.set_page_config(page_title="Prediksi Diabetes LSTM", layout="centered")
st.title("🩺 Prediksi Diabetes dengan LSTM")
st.write("Silakan masukkan data pasien untuk prediksi.")

# Form input
with st.form("form_prediksi"):
    pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Kadar Glukosa", min_value=0, max_value=300)
    blood_pressure = st.number_input("Tekanan Darah", min_value=0, max_value=200)
    skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0, max_value=100)
    insulin = st.number_input("Kadar Insulin", min_value=0, max_value=900)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)
    age = st.number_input("Usia", min_value=1, max_value=120)

    submitted = st.form_submit_button("Prediksi")

# Prediksi
if submitted:
    try:
        # Load scaler dan model LSTM
        scaler = joblib.load("scaler.pkl")
        model = load_model("model_lstm.h5")

        # Format input & scaling
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        # Reshape ke (samples, timesteps, features)
        input_reshaped = input_scaled.reshape((1, input_scaled.shape[1], 1))  # shape: (1, 8, 1)

        # Lakukan prediksi
        prediction = model.predict(input_reshaped)

        # Tampilkan hasil
        if prediction[0][0] >= 0.5:
            st.error(f"⚠️ Pasien diprediksi berisiko diabetes (score: {prediction[0][0]:.2f})")
        else:
            st.success(f"✅ Pasien tidak berisiko diabetes (score: {prediction[0][0]:.2f})")

    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: {e.filename}")
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
