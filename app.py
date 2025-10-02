import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Ganti dari pickle ke joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- LOAD PRETRAINED MODEL AND SCALER ---
MODEL_PATH = "materials/svr_coal_model.pkl"
SCALER_PATH = "materials/scaler_coal.pkl"
DATA_PATH = "materials/Combined_modelling.xlsx"

# Load SVR model
svr_model = joblib.load(MODEL_PATH)

# Load Scaler
scaler = joblib.load(SCALER_PATH)

# --- LOAD DATA ---
@st.cache
def load_data():
    df = pd.read_excel(DATA_PATH)
    df = df[df["Jenis_Energi"] == "Batu Bara"].drop(columns=["Tipe_Aliran"])
    df = df.T.reset_index()
    df.columns = ['Tahun', 'Produksi']
    df = df[1:]
    df["Tahun"] = pd.to_datetime(df["Tahun"].astype(int), format='%Y')
    df["Produksi"] = df["Produksi"].astype(float)
    df = df.set_index("Tahun")
    df = df.resample("MS").interpolate(method="linear")  # Resample ke bulanan
    return df

df = load_data()

# --- SMOOTHING FUNCTION ---
def moving_average_smoothing(series, window=6):
    """Fungsi untuk melakukan smoothing menggunakan moving average."""
    return pd.Series(series).rolling(window=window, center=True, min_periods=1).mean()

# --- STREAMLIT APP ---
st.title("Prediksi Produksi Energi Batu Bara")
st.write("Aplikasi ini memprediksi produksi energi batu bara berdasarkan model SVR yang telah dilatih.")

# --- USER INPUT ---
st.sidebar.header("Input Prediksi")
target_year = st.sidebar.number_input("Masukkan Tahun Prediksi", min_value=2024, max_value=2050, value=2030, step=1)

# --- FORECAST FUNCTION ---
def forecast_production(year, model, scaler, data):
    # Hitung jumlah bulan dari data terakhir hingga target tahun
    last_date = data.index[-1]
    target_date = pd.to_datetime(f"{year}-12-01")
    future_months = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), end=target_date, freq="MS")

    # Ambil 12 nilai terakhir sebagai input awal
    recent_values = data["Produksi"].values[-12:].tolist()
    future_preds = []

    for _ in range(len(future_months)):
        # Buat input lag 12 bulan
        lag_input = np.array(recent_values[-12:]).reshape(1, -1)
        lag_scaled = scaler.transform(lag_input)
        # Prediksi menggunakan model
        pred = model.predict(lag_scaled)[0]
        future_preds.append(pred)
        recent_values.append(pred)

    # Gabungkan prediksi dengan index waktu
    future_df = pd.DataFrame({
        "Tahun": future_months,
        "Produksi": future_preds
    }).set_index("Tahun")

    # Terapkan smoothing pada hasil prediksi
    future_df["Produksi"] = moving_average_smoothing(future_df["Produksi"])

    return future_df

# --- RUN FORECAST ---
future_df = forecast_production(target_year, svr_model, scaler, df)

# --- DISPLAY RESULTS ---
try:
    # Ambil hasil prediksi untuk bulan Desember di tahun target
    december_prediction = future_df.loc[future_df.index.month == 12].loc[f"{target_year}-12", "Produksi"]

    # Pastikan hasil prediksi adalah numerik
    if pd.notnull(december_prediction):
        st.subheader("Hasil Prediksi")
        st.write(f"Prediksi Produksi Energi Batu Bara untuk Bulan Desember Tahun {target_year}:")
        st.write(f"{december_prediction:.2f}")
    else:
        st.subheader("Hasil Prediksi")
        st.write(f"Tidak ada data prediksi untuk Bulan Desember Tahun {target_year}.")
except KeyError:
    st.subheader("Hasil Prediksi")
    st.write(f"Tidak ada data prediksi untuk Bulan Desember Tahun {target_year}.")
except Exception as e:
    st.subheader("Hasil Prediksi")
    st.write(f"Terjadi error: {e}")

# --- PLOT RESULTS ---
import matplotlib.pyplot as plt

st.subheader("Visualisasi Prediksi")
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Produksi"], label="Data Aktual", color="blue")
plt.plot(future_df.index, future_df["Produksi"], label="Prediksi (Smoothed)", color="red", linestyle="--")
plt.axvline(x=pd.to_datetime("2024-01-01"), color="black", linestyle=":", label="Awal Prediksi")
plt.title("Prediksi Produksi Energi Batu Bara")
plt.xlabel("Tahun")
plt.ylabel("Produksi")
plt.grid(True)
plt.legend()
st.pyplot(plt)
