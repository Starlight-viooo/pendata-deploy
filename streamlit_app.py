import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from ucimlrepo import fetch_ucirepo

st.set_page_config(page_title="Prediksi Penyakit Hati", layout="wide")
st.title("🧬 Prediksi Penyakit Hati (Liver Disease Classifier)")
st.caption("Model menggunakan kombinasi GaussianNB dan CategoricalNB")

# ============================
# Fungsi Pelatihan & Preprocessing
# ============================

@st.cache_resource
def train_model():
    data = fetch_ucirepo(id=225)
    X = data.data.features.copy()
    y = data.data.targets.copy()

    # Outlier treatment
    for col in X.select_dtypes(include=["float64", "int64"]):
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mean_val = X[(X[col] >= lower) & (X[col] <= upper)][col].mean()
        X[col] = X[col].apply(lambda x: mean_val if x < lower or x > upper else x)

    # Tangani missing value di A/G Ratio
    X['A/G Ratio'] = X['A/G Ratio'].fillna(X['A/G Ratio'].median())

    # Encode Gender
    le = LabelEncoder()
    X['Gender'] = le.fit_transform(X['Gender'])  # 0 = Female, 1 = Male

    # Imputasi numerik lainnya
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    y = y.values.ravel()

    # Pisahkan fitur
    X_num = X.drop(columns=['Gender'])
    X_cat = X[['Gender']]
    
    # Scaling numerik
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Train models
    gnb = GaussianNB().fit(X_num_scaled, y)
    cnb = CategoricalNB().fit(X_cat, y)

    return gnb, cnb, scaler, le

# Inisialisasi model dan scaler
gnb, cnb, scaler, label_encoder = train_model()

# ============================
# Antarmuka Input Sidebar
# ============================

st.sidebar.header("🧾 Input Data Pasien")

def user_input():
    Age = st.sidebar.slider("Usia", 10, 100, 50)
    Gender_str = st.sidebar.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    Gender = 1 if Gender_str == "Laki-laki" else 0
    TB = st.sidebar.number_input("Bilirubin Total (TB)", 0.0, 75.0, 1.5, 0.1)
    DB = st.sidebar.number_input("Bilirubin Langsung (DB)", 0.0, 25.0, 0.4, 0.1)
    Alkphos = st.sidebar.number_input("Alkaline Phosphotase", 50, 2200, 200)
    Sgpt = st.sidebar.number_input("SGPT", 10, 2000, 45)
    Sgot = st.sidebar.number_input("SGOT", 10, 3000, 50)
    TP = st.sidebar.number_input("Total Protein", 2.0, 10.0, 6.5)
    ALB = st.sidebar.number_input("Albumin", 0.5, 6.5, 3.2)
    AGR = st.sidebar.number_input("Albumin/Globulin Ratio", 0.1, 3.0, 1.0)

    return pd.DataFrame({
        "Age": [Age],
        "TB": [TB],
        "DB": [DB],
        "Alkphos": [Alkphos],
        "Sgpt": [Sgpt],
        "Sgot": [Sgot],
        "TP": [TP],
        "ALB": [ALB],
        "A/G Ratio": [AGR],
        "Gender": [Gender]
    })

input_df = user_input()

st.subheader("📋 Data yang Dimasukkan")
st.dataframe(input_df)

# ============================
# Prediksi
# ============================

if st.button("🔮 Prediksi Sekarang"):
    input_num = input_df.drop(columns=["Gender"])
    input_cat = input_df[["Gender"]]
    input_num_scaled = scaler.transform(input_num)

    log_prob_gnb = gnb.predict_log_proba(input_num_scaled)
    log_prob_cnb = cnb.predict_log_proba(input_cat)
    combined_log_prob = log_prob_gnb + log_prob_cnb

    prediction = gnb.classes_[np.argmax(combined_log_prob, axis=1)][0]

    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("⚠️ Pasien TERINDIKASI memiliki penyakit hati.")
    else:
        st.success("✅ Pasien TIDAK terindikasi memiliki penyakit hati.")

