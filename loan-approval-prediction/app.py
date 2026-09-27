import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Melatih model SVM secara instan dan di-cache
@st.cache_resource
def get_model():
    data = pd.read_csv('loan.csv').dropna()
    data = data.replace(to_replace='3+', value=4)
    data.replace({
        'Married': {'No': 0, 'Yes': 1},
        'Gender': {'Female': 0, 'Male': 1},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Education': {'Not Graduate': 0, 'Graduate': 1},
        'Loan_Status': {'N': 0, 'Y': 1}
    }, inplace=True)
    data['Dependents'] = data['Dependents'].astype(int)
    
    X = data.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
    y = data['Loan_Status']
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', svm.SVC(kernel='linear', random_state=2))
    ])
    pipeline.fit(X, y)
    return pipeline, list(X.columns)

model, feature_names = get_model()

# Header Aplikasi
st.title("Prediksi Kelayakan Pinjaman Bank (Loan Approval)")
st.write("Masukkan profil nasabah untuk memprediksi apakah pengajuan kredit disetujui atau ditolak:")

# Form Input Data Nasabah
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    married = st.selectbox("Status Pernikahan", ["Sudah Menikah", "Lajang"])
    dependents = st.selectbox("Jumlah Tanggungan", ["0", "1", "2", "3+"])
    education = st.selectbox("Pendidikan Terakhir", ["Sarjana (Graduate)", "Non-Sarjana (Not Graduate)"])
    self_employed = st.selectbox("Status Pekerjaan", ["Karyawan / Profesional", "Wirausaha (Self Employed)"])

with col2:
    applicant_income = st.number_input("Pendapatan Pemohon ($ / Bulan)", value=4500, step=100)
    coapplicant_income = st.number_input("Pendapatan Pasangan ($ / Bulan)", value=1500, step=100)
    loan_amount = st.number_input("Jumlah Pinjaman ($ Ribu, contoh: 120 = $120.000)", value=128, step=10)
    loan_term = st.number_input("Jangka Waktu Pinjaman (Bulan)", value=360, step=12)
    credit_history = st.selectbox("Riwayat Kredit", ["Lancar / Memenuhi Syarat (1.0)", "Ada Tunggakan / Tidak Lolos (0.0)"])
    property_area = st.selectbox("Lokasi Properti", ["Perkotaan (Urban)", "Pinggiran Kota (Semiurban)", "Pedesaan (Rural)"])

# Konversi input ke format model
gender_val = 1 if gender == "Laki-laki" else 0
married_val = 1 if married == "Sudah Menikah" else 0
dep_val = 4 if dependents == "3+" else int(dependents)
edu_val = 1 if "Sarjana" in education else 0
emp_val = 1 if "Wirausaha" in self_employed else 0
cred_val = 1.0 if "1.0" in credit_history else 0.0

prop_map = {"Pedesaan (Rural)": 0, "Pinggiran Kota (Semiurban)": 1, "Perkotaan (Urban)": 2}
prop_val = prop_map[property_area]

# Tombol Prediksi
if st.button("Cek Kelayakan Pinjaman"):
    input_df = pd.DataFrame([[
        gender_val, married_val, dep_val, edu_val, emp_val,
        applicant_income, coapplicant_income, loan_amount,
        loan_term, cred_val, prop_val
    ]], columns=feature_names)
    
    keputusan = model.predict(input_df)[0]
    
    if keputusan == 1:
        st.success("SELAMAT! Pengajuan Pinjaman DISETUJUI (Approved)")
        st.write("Profil finansial dan riwayat kredit nasabah memenuhi standar kelayakan kredit perbankan.")
    else:
        st.error("MAAF, Pengajuan Pinjaman DITOLAK (Rejected)")
        st.write("Profil risiko nasabah belum memenuhi standar kelayakan bank (periksa kembali riwayat kredit atau rasio pinjaman).")
