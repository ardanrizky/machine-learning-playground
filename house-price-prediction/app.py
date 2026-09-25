import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Melatih model secara instan dan di-cache
@st.cache_resource
def get_model():
    data = pd.read_csv('housing.csv')
    X = data.drop(columns='MedHouseVal')
    y = data['MedHouseVal']
    model = XGBRegressor(random_state=2)
    model.fit(X, y)
    return model

model = get_model()

# Deskripsi Web
st.title("Prediksi Harga Rumah California")
st.write("Masukkan karakteristik properti untuk melihat taksiran harga:")

# Form Input Data Rumah
med_inc = st.number_input("Pendapatan Rata-rata Wilayah (MedInc) x $10.000", value=8.32)
house_age = st.number_input("Usia Bangunan Rumah (Tahun)", value=41.0)
ave_rooms = st.number_input("Rata-rata Jumlah Kamar", value=6.98)
ave_bedrms = st.number_input("Rata-rata Jumlah Kamar Tidur", value=1.02)
population = st.number_input("Jumlah Populasi Wilayah", value=322.0)
ave_occup = st.number_input("Rata-rata Penghuni per Rumah", value=2.55)
latitude = st.number_input("Garis Lintang (Latitude)", value=37.88)
longitude = st.number_input("Garis Bujur (Longitude)", value=-122.23)

# Prediksi
if st.button("Hitung Estimasi Harga"):
    data_input = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    taksiran = model.predict(data_input)[0]
    
    harga_dollar = taksiran * 100000
    harga_rupiah = harga_dollar * 16000
    
    st.success("Taksiran Berhasil Dihitung!")
    st.write(f"Taksiran Harga (USD): ${harga_dollar:,.2f}")
    st.write(f"Setara Rupiah: Rp {harga_rupiah:,.2f}")
