import streamlit as st
import joblib
import numpy as np

# Judul halaman
st.set_page_config(page_title="Prediksi Diabetes", layout="centered")
st.title("🩺 Prediksi Diabetes")
st.write("Silakan masukkan data pasien untuk prediksi.")

# Formulir input
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

# Proses prediksi
if submitted:
    try:
        # Load model
        model = joblib.load("model_diabetes.pkl")

        # Siapkan data input
        data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

        # Prediksi
        pred = model.predict(data)

        # Tampilkan hasil
        if pred[0] == 1:
            st.error("⚠️ Pasien diprediksi memiliki risiko diabetes.")
        else:
            st.success("✅ Pasien tidak memiliki risiko diabetes.")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
