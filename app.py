import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Ganti dari pickle ke joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- PATHS ---
DATA_PATH = "materials/Combined_modelling.xlsx"
MODEL_PATHS = {
    "Batu Bara": "materials/svr_coal_model.pkl",
    "Gas Alam": "materials/svr_natural_gas_model.pkl"
}
SCALER_PATHS = {
    "Batu Bara": "materials/scaler_coal.pkl",
    "Gas Alam": "materials/scaler_natural_gas.pkl"
}

# --- LOAD DATA ---
@st.cache
def load_data(energy_type):
    """Load dan filter data berdasarkan jenis energi."""
    df = pd.read_excel(DATA_PATH)
    df = df[df["Jenis_Energi"] == energy_type].drop(columns=["Tipe_Aliran"])
    df = df.T.reset_index()
    df.columns = ['Tahun', 'Produksi']
    df = df[1:]
    df["Tahun"] = pd.to_datetime(df["Tahun"].astype(int), format='%Y')
    df["Produksi"] = df["Produksi"].astype(float)
    df = df.set_index("Tahun")
    df = df.resample("MS").interpolate(method="linear")  # Resample ke bulanan
    return df

# --- SMOOTHING FUNCTION ---
def moving_average_smoothing(series, window=6):
    """Fungsi untuk melakukan smoothing menggunakan moving average."""
    return pd.Series(series).rolling(window=window, center=True, min_periods=1).mean()

# --- FORECAST FUNCTION ---
def forecast_production(year, model, scaler, data):
    """Melakukan prediksi hingga bulan Desember tahun target."""
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

# --- STREAMLIT APP ---
st.title("Prediksi Produksi Energi")
st.write("Aplikasi ini memprediksi produksi energi berdasarkan model SVR yang telah dilatih.")

# --- USER INPUT ---
st.sidebar.header("Input Prediksi")
energy_type = st.sidebar.selectbox("Pilih Jenis Energi", options=["Batu Bara", "Gas Alam"])
target_year = st.sidebar.number_input("Masukkan Tahun Prediksi", min_value=2024, max_value=2050, value=2030, step=1)

# --- LOAD MODEL & SCALER ---
try:
    model_path = MODEL_PATHS[energy_type]
    scaler_path = SCALER_PATHS[energy_type]

    # Load model dan scaler
    svr_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load data
    df = load_data(energy_type)

    # --- RUN FORECAST ---
    future_df = forecast_production(target_year, svr_model, scaler, df)

    # --- DISPLAY RESULTS ---
    try:
        # Ambil hasil prediksi untuk bulan Desember di tahun target
        december_prediction = future_df.loc[f"{target_year}-12", "Produksi"]

        # Pastikan hasil prediksi adalah scalar
        if not pd.isnull(december_prediction):
            st.subheader("Hasil Prediksi")
            st.write(f"Prediksi Produksi Energi {energy_type} untuk Bulan Desember Tahun {target_year}:")
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
    st.subheader("Visualisasi Prediksi")
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Produksi"], label="Data Aktual", color="blue")
    plt.plot(future_df.index, future_df["Produksi"], label="Prediksi (Smoothed)", color="red", linestyle="--")
    plt.axvline(x=pd.to_datetime("2024-01-01"), color="black", linestyle=":", label="Awal Prediksi")
    plt.title(f"Prediksi Produksi Energi {energy_type}")
    plt.xlabel("Tahun")
    plt.ylabel("Produksi")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
except Exception as e:
    st.subheader("Error")
    st.write(f"Terjadi error saat memuat model atau data: {e}")
