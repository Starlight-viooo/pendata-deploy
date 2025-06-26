import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from ucimlrepo import fetch_ucirepo

# =================================================================================
# BAGIAN 1: MELATIH MODEL DAN MENYIAPKAN PREPROCESSOR (MENGGUNAKAN CACHING)
# =================================================================================

# st.cache_resource akan menjalankan fungsi ini sekali saja dan menyimpan hasilnya.
# Ini mencegah model dilatih ulang setiap kali pengguna berinteraksi dengan aplikasi.
@st.cache_resource
def train_model_and_get_preprocessors():
    """
    Fungsi ini melakukan semua langkah preprocessing dan training model.
    Mengembalikan model yang sudah dilatih dan preprocessor yang diperlukan.
    """
    # 1. Mengunduh dataset
    ilpd = fetch_ucirepo(id=225)
    X = ilpd.data.features
    y = ilpd.data.targets
    
    # 2. Pembersihan Outlier (Sama seperti logika Anda)
    X_clean = X.copy()
    numeric_cols = X_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        Q1 = X_clean[col].quantile(0.25)
        Q3 = X_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        non_outliers = X_clean[(X_clean[col] >= lower) & (X_clean[col] <= upper)][col]
        mean_normal = non_outliers.mean()
        X_clean[col] = X_clean[col].apply(lambda x: mean_normal if x < lower or x > upper else x)
    
    # 3. Imputasi dan Encoding
    # Tangani nilai NaN pada 'A/G Ratio' sebelum encoding
    X_clean['A/G Ratio'] = X_clean['A/G Ratio'].fillna(X_clean['A/G Ratio'].median())
    
    # Encode 'Gender'
    le = LabelEncoder()
    X_clean['Gender'] = le.fit_transform(X_clean['Gender']) # 0 untuk Female, 1 untuk Male

    # 4. Memisahkan Fitur
    X_num = X_clean.drop(columns=['Gender'])
    X_cat = X_clean[['Gender']]
    y_flat = y.values.ravel()

    # 5. Scaling Fitur Numerik
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # 6. Melatih Model (tanpa train_test_split, kita latih pada semua data)
    # Model dilatih pada semua data yang tersedia agar lebih robust.
    # train_test_split hanya untuk evaluasi, bukan untuk produk final.
    gnb = GaussianNB()
    gnb.fit(X_num_scaled, y_flat)

    cnb = CategoricalNB()
    cnb.fit(X_cat, y_flat)

    # Mengembalikan semua objek yang kita butuhkan untuk prediksi nanti
    return gnb, cnb, scaler, le, X_clean.columns

# Panggil fungsi di atas untuk memuat model dan preprocessor
gnb_model, cnb_model, scaler, label_encoder, feature_columns = train_model_and_get_preprocessors()


# =================================================================================
# BAGIAN 2: TAMPILAN ANTARMUKA APLIKASI STREAMLIT (FRONT-END)
# =================================================================================

st.set_page_config(page_title="Prediksi Penyakit Hati", layout="wide")
st.title("👨‍⚕️ Aplikasi Prediksi Penyakit Hati")
st.write("Aplikasi ini menggunakan model *Mixed Naive Bayes* untuk memprediksi kemungkinan seseorang memiliki penyakit hati berdasarkan data klinis. Silakan masukkan data pasien di bawah ini.")

st.sidebar.header("Input Data Pasien")

# Fungsi untuk membuat input dari pengguna di sidebar
def user_input_features():
    age = st.sidebar.slider('Usia (Age)', 1, 100, 45)
    gender_str = st.sidebar.radio('Jenis Kelamin (Gender)', ('Laki-laki (Male)', 'Perempuan (Female)'))
    tb = st.sidebar.number_input('Bilirubin Total (TB)', 0.1, 80.0, 1.0, 0.1)
    db = st.sidebar.number_input('Bilirubin Langsung (DB)', 0.1, 20.0, 0.5, 0.1)
    alkphos = st.sidebar.number_input('Alkaline Phosphotase (Alkphos)', 50, 2200, 200, 10)
    sgpt = st.sidebar.number_input('Alamine Aminotransferase (Sgpt)', 10, 2000, 50, 10)
    sgot = st.sidebar.number_input('Aspartate Aminotransferase (Sgot)', 10, 5000, 50, 10)
    tp = st.sidebar.number_input('Total Protein (TP)', 2.0, 10.0, 6.5, 0.1)
    alb = st.sidebar.number_input('Albumin (ALB)', 0.5, 6.0, 3.5, 0.1)
    ag_ratio = st.sidebar.number_input('Rasio Albumin/Globulin (A/G Ratio)', 0.1, 3.0, 1.0, 0.1)

    # Konversi gender ke format numerik yang sesuai dengan LabelEncoder
    gender_num = 1 if gender_str == 'Laki-laki (Male)' else 0
    
    data = {
        'Age': age,
        'Gender': gender_num,
        'TB': tb,
        'DB': db,
        'Alkphos': alkphos,
        'Sgpt': sgpt,
        'Sgot': sgot,
        'TP': tp,
        'ALB': alb,
        'A/G Ratio': ag_ratio
    }
    
    # Membuat DataFrame dengan urutan kolom yang benar
    features = pd.DataFrame(data, index=[0])
    
    # Pastikan urutan kolom sama persis dengan saat training
    feature_order = [col for col in feature_columns if col != 'Selector']
    return features[feature_order]

# Ambil input dari user
input_df = user_input_features()

# Tampilkan data yang diinput pengguna
st.subheader("Data Pasien yang Dimasukkan:")
st.dataframe(input_df)

# =================================================================================
# BAGIAN 3: MEMBUAT PREDIKSI DAN MENAMPILKAN HASIL
# =================================================================================

# Tombol untuk memicu prediksi
if st.button("🔮 Buat Prediksi", key='predict_button'):
    # 1. Pisahkan fitur numerik dan kategorikal dari input
    input_num = input_df.drop(columns=['Gender'])
    input_cat = input_df[['Gender']]

    # 2. Scaling fitur numerik menggunakan scaler yang sudah di-fit
    input_num_scaled = scaler.transform(input_num)

    # 3. Membuat prediksi menggunakan model yang sudah dilatih
    log_prob_gnb = gnb_model.predict_log_proba(input_num_scaled)
    log_prob_cnb = cnb_model.predict_log_proba(input_cat)

    # Gabungkan probabilitas logaritmik
    combined_log_prob = log_prob_gnb + log_prob_cnb
    
    # Ambil kelas dengan probabilitas tertinggi
    # Kelas asli: 1=Sakit, 2=Tidak Sakit
    prediction = gnb_model.classes_[np.argmax(combined_log_prob, axis=1)][0]

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("⚠️ Hasil: Pasien **TERINDIKASI** memiliki penyakit hati.", icon="🚨")
        st.write("Disarankan untuk melakukan konsultasi lebih lanjut dengan dokter.")
    else:
        st.success("✅ Hasil: Pasien **TIDAK TERINDIKASI** memiliki penyakit hati.", icon="👍")
        st.write("Tetap jaga pola hidup sehat.")
