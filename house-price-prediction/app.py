import streamlit as st
import joblib
import numpy as np

#Load model 
model = joblib.load('model_harga_rumah.sav')

#Deskripsi Web
st.title("🏠 Prediksi Harga Rumah California")
st.write("Masukkan karakteristik properti untuk melihat taksiran harga:")

#Form Input Data Rumah
med_inc = st.number_input("Pendapatan Rata-rata Wilayah (MedInc) x $10.000", value=8.32)
house_age = st.number_input("Usia Bangunan Rumah (Tahun)", value=41.0)
ave_rooms = st.number_input("Rata-rata Jumlah Kamar", value=6.98)
ave_bedrms = st.number_input("Rata-rata Jumlah Kamar Tidur", value=1.02)
population = st.number_input("Jumlah Populasi Wilayah", value=322.0)
ave_occup = st.number_input("Rata-rata Penghuni per Rumah", value=2.55)
latitude = st.number_input("Garis Lintang (Latitude)", value=37.88)
longitude = st.number_input("Garis Bujur (Longitude)", value=-122.23)

#Prediksi
if st.button("Hitung Estimasi Harga"):
    # Susun data input menjadi array
    data_input = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    
    # Lakukan prediksi dengan model
    taksiran = model.predict(data_input)[0]
    
    # Hitung konversi ke USD dan Rupiah
    harga_dollar = taksiran * 100000
    harga_rupiah = harga_dollar * 16000
    
    # Tampilkan hasil di layar
    st.success("Taksiran Berhasil Dihitung!")
    st.write(f" **Taksiran Harga (USD):** ${harga_dollar:,.2f}")
    st.write(f"🇮🇩 **Setara Rupiah:** Rp {harga_rupiah:,.2f}")
