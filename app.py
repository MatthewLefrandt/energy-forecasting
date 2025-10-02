import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- PATHS ---
DATA_PATH = "materials/Combined_modelling.xlsx"
MODEL_PATHS = {
    "Batu Bara": "materials/svr_coal_model.pkl",
    "Gas Alam": "materials/svr_natural_gas_model.pkl",
    "Minyak Bumi": "materials/svr_petroleum_model.pkl",
    "Biodiesel": "materials/linreg_biodiesel_model.pkl"
}
SCALER_PATHS = {
    "Batu Bara": "materials/scaler_coal.pkl",
    "Gas Alam": "materials/scaler_natural_gas.pkl",
    "Minyak Bumi": "materials/scaler_petroleum.pkl",
    "Biodiesel": "materials/scaler_biodiesel.pkl"
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
    df["Tahun"] = df["Tahun"].astype(int)
    df["Produksi"] = pd.to_numeric(df["Produksi"], errors='coerce')  # Pastikan semua nilai numerik
    df = df.set_index("Tahun")
    df = df.interpolate(method="linear")  # Mengisi nilai kosong dengan interpolasi linear
    df = df.dropna()  # Pastikan tidak ada nilai kosong setelah interpolasi
    return df

# --- SMOOTHING FUNCTION ---
def moving_average_smoothing(series, window=3):
    """Fungsi untuk melakukan smoothing menggunakan moving average."""
    return pd.Series(series).rolling(window=window, center=True, min_periods=1).mean()

# --- FORECAST FUNCTION ---
def forecast_production(year, model, scaler, data, energy_type):
    """Melakukan prediksi hingga bulan Desember tahun target."""
    if energy_type == "Biodiesel":
        # Prediksi tahunan untuk Biodiesel
        last_year = data.index.max().year
        future_years = list(range(last_year + 1, year + 1))
        recent_value = data["Produksi"].values[-1]
        future_preds = []

        for _ in future_years:
            lag_input = np.array([[recent_value]])
            lag_scaled = scaler.transform(lag_input)
            pred = model.predict(lag_scaled)[0]
            future_preds.append(pred)
            recent_value = pred  # update untuk tahun berikutnya

        # Gabungkan prediksi dengan index waktu
        future_df = pd.DataFrame({
            "Tahun": future_years,
            "Produksi": future_preds
        }).set_index("Tahun")

        # Terapkan smoothing pada hasil prediksi
        future_df["Produksi"] = moving_average_smoothing(future_df["Produksi"])
        return future_df
    else:
        # Prediksi bulanan untuk Batu Bara, Gas Alam, dan Minyak Bumi
        last_date = data.index[-1]
        target_date = pd.to_datetime(f"{year}-12-01")
        future_months = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), end=target_date, freq="MS")

        # Ambil 12 nilai terakhir sebagai input awal
        recent_values = data["Produksi"].values[-12:].tolist()
        future_preds = []

        for _ in range(len(future_months)):
            lag_input = np.array(recent_values[-12:]).reshape(1, -1)
            lag_scaled = scaler.transform(lag_input)
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
st.write("Aplikasi ini memprediksi produksi energi berdasarkan model yang telah dilatih.")

# --- USER INPUT ---
st.sidebar.header("Input Prediksi")
energy_type = st.sidebar.selectbox("Pilih Jenis Energi", options=["Batu Bara", "Gas Alam", "Minyak Bumi", "Biodiesel"])
target_year = st.sidebar.number_input("Masukkan Tahun Prediksi", min_value=2025, max_value=2100, value=2025, step=1)

# --- LOAD MODEL & SCALER ---
try:
    model_path = MODEL_PATHS[energy_type]
    scaler_path = SCALER_PATHS[energy_type]

    # Load model dan scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load data
    df = load_data(energy_type)

    # --- RUN FORECAST ---
    future_df = forecast_production(target_year, model, scaler, df, energy_type)

    # --- DISPLAY RESULTS ---
    try:
        if energy_type == "Biodiesel":
            # Ambil hasil prediksi untuk tahun target
            prediction = future_df.loc[target_year, "Produksi"]
        else:
            # Ambil hasil prediksi untuk bulan Desember di tahun target
            prediction = future_df.loc[f"{target_year}-12", "Produksi"]

        # Pastikan hasil prediksi adalah scalar
        if isinstance(prediction, pd.Series):
            prediction = prediction.iloc[0]

        if not pd.isnull(prediction):
            st.subheader("Hasil Prediksi")
            st.write(f"Prediksi Produksi Energi {energy_type} untuk Tahun {target_year}:")
            st.write(f"{prediction:.2f}")
        else:
            st.subheader("Hasil Prediksi")
            st.write(f"Tidak ada data prediksi untuk Tahun {target_year}.")
    except KeyError:
        st.subheader("Hasil Prediksi")
        st.write(f"Tidak ada data prediksi untuk Tahun {target_year}.")
    except Exception as e:
        st.subheader("Hasil Prediksi")
        st.write(f"Terjadi error: {e}")

    # --- PLOT RESULTS ---
    st.subheader("Visualisasi Prediksi")
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Produksi"], label="Data Aktual", color="blue")
    plt.plot(future_df.index, future_df["Produksi"], label="Prediksi (Smoothed)", color="red", linestyle="--")
    plt.axvline(x=pd.to_datetime(f"{target_year}-01-01"), color="black", linestyle=":", label="Awal Prediksi")
    plt.title(f"Prediksi Produksi Energi {energy_type}")
    plt.xlabel("Tahun")
    plt.ylabel("Produksi")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
except Exception as e:
    st.subheader("Error")
    st.write(f"Terjadi error saat memuat model atau data: {e}")
