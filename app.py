import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt

# --- PATHS ---
DATA_PATH = "materials/Combined_modelling.xlsx"
MODEL_PATHS = {
    "Batu Bara": "materials/svr_coal_model.pkl",
    "Gas Alam": "materials/svr_natural_gas_model.pkl",
    "Minyak Bumi": "materials/svr_petroleum_model.pkl",
    "Biodiesel": "materials/linreg_biodiesel_model.pkl",
    "Fuel Ethanol": "materials/svr_fuel_ethanol_model.pkl"  # Tambahkan model Fuel Ethanol
}
SCALER_PATHS = {
    "Batu Bara": "materials/scaler_coal.pkl",
    "Gas Alam": "materials/scaler_natural_gas.pkl",
    "Minyak Bumi": "materials/scaler_petroleum.pkl",
    "Biodiesel": "materials/scaler_biodiesel.pkl",
    "Fuel Ethanol": "materials/scaler_fuel_ethanol.pkl"  # Tambahkan scaler Fuel Ethanol
}

# --- LOAD DATA ---
@st.cache_data
def load_data(energy_type):
    """Load dan filter data berdasarkan jenis energi."""
    df = pd.read_excel(DATA_PATH)
    df = df[df["Jenis_Energi"] == energy_type].drop(columns=["Tipe_Aliran"])
    df = df.T.reset_index()
    df.columns = ['Tahun', 'Produksi']
    df = df[1:]  # Hapus baris header

    # Jika Biodiesel, gunakan data tahunan dan tangani missing values
    if energy_type == "Biodiesel":
        df["Tahun"] = df["Tahun"].astype(int)
        df["Produksi"] = df["Produksi"].astype(float)
        # Hapus data dengan nilai NaN atau 0
        df = df.dropna()
        df = df[df["Produksi"] > 0]
        df = df.set_index("Tahun")
        return df
    else:
        # Untuk energi lain, gunakan konversi ke data bulanan
        df["Tahun"] = pd.to_datetime(df["Tahun"].astype(int), format='%Y')
        df["Produksi"] = df["Produksi"].astype(float)
        df = df.set_index("Tahun")
        df = df.resample("MS").interpolate(method="linear")  # Resample ke bulanan
        return df

# --- SMOOTHING FUNCTION ---
def moving_average_smoothing(series, window=6):
    """Fungsi untuk melakukan smoothing menggunakan moving average."""
    return pd.Series(series).rolling(window=window, center=True, min_periods=1).mean()

# --- FORECAST FUNCTION FOR SVR MODELS (MONTHLY) ---
def forecast_production_svr(year, model, scaler, data):
    """Melakukan prediksi hingga bulan Desember tahun target untuk model SVR."""
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

# --- FORECAST FUNCTION FOR LINEAR REGRESSION (YEARLY) ---
def forecast_production_biodiesel(year, model, scaler, data):
    """Melakukan prediksi hingga tahun target untuk model Linear Regression Biodiesel."""
    # Ambil tahun terakhir dan nilai produksi terakhir
    last_year = data.index[-1]
    last_value = data["Produksi"].values[-1]

    # Hitung tahun yang perlu diprediksi
    future_years = list(range(last_year + 1, year + 1))

    if not future_years:  # Jika tahun target <= tahun terakhir data
        return pd.DataFrame()

    # Lakukan prediksi tahun demi tahun
    future_preds = []
    recent_value = last_value

    for _ in future_years:
        # Buat input (lag 1 tahun)
        lag_input = np.array([[recent_value]])
        lag_scaled = scaler.transform(lag_input)
        # Prediksi menggunakan model
        pred = model.predict(lag_scaled)[0]
        future_preds.append(pred)
        recent_value = pred  # Update untuk tahun berikutnya

    # Gabungkan prediksi dengan index tahun
    future_df = pd.DataFrame({
        "Tahun": future_years,
        "Produksi": future_preds
    }).set_index("Tahun")

    # Terapkan smoothing pada hasil prediksi
    future_df["Produksi"] = moving_average_smoothing(future_df["Produksi"], window=3)

    return future_df

# --- STREAMLIT APP ---
st.title("Prediksi Produksi Energi")
st.write("Aplikasi ini memprediksi produksi energi berdasarkan model yang telah dilatih.")

# --- USER INPUT ---
st.sidebar.header("Input Prediksi")
energy_type = st.sidebar.selectbox("Pilih Jenis Energi", 
                                  options=["Batu Bara", "Gas Alam", "Minyak Bumi", "Biodiesel", "Fuel Ethanol"])
target_year = st.sidebar.number_input("Masukkan Tahun Prediksi", min_value=2025, max_value=2100, value=2030, step=1)

# --- LOAD MODEL & SCALER ---
try:
    model_path = MODEL_PATHS[energy_type]
    scaler_path = SCALER_PATHS[energy_type]

    # Load model dan scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load data
    df = load_data(energy_type)

    # Informasi tentang data yang digunakan
    if energy_type == "Biodiesel":
        st.info(f"Data {energy_type} tersedia dari tahun {df.index.min()} hingga {df.index.max()}. Data sebelum tahun 2000 tidak tersedia.")
    elif energy_type == "Fuel Ethanol":
        st.info(f"Data {energy_type} tersedia dari tahun 1980 hingga 2022.")

    # --- RUN FORECAST ---
    if energy_type == "Biodiesel":
        # Tambahkan validasi: pastikan df tidak kosong
        if df.empty:
            st.error("Data Biodiesel tidak cukup untuk melakukan prediksi. Tidak ada data yang tersedia.")
        else:
            # Tambahkan Lag_1 ke data untuk Biodiesel (diperlukan untuk model)
            df["Lag_1"] = df["Produksi"].shift(1)
            df = df.dropna()  # Hapus baris dengan NaN setelah shifting

            # Jalankan prediksi jika data tidak kosong
            if not df.empty:
                future_df = forecast_production_biodiesel(target_year, model, scaler, df)

                # --- DISPLAY RESULTS ---
                try:
                    # Ambil hasil prediksi untuk tahun target
                    if target_year in future_df.index:
                        year_prediction = future_df.loc[target_year, "Produksi"]

                        # Pastikan hasil prediksi adalah scalar
                        if isinstance(year_prediction, pd.Series):
                            year_prediction = year_prediction.iloc[0]

                        if not pd.isnull(year_prediction):
                            st.subheader("Hasil Prediksi")
                            st.write(f"Prediksi Produksi Energi {energy_type} untuk Tahun {target_year}:")
                            st.write(f"{year_prediction:.2f}")
                        else:
                            st.subheader("Hasil Prediksi")
                            st.write(f"Tidak ada data prediksi untuk Tahun {target_year}.")
                    else:
                        st.subheader("Hasil Prediksi")
                        st.write(f"Tidak ada data prediksi untuk Tahun {target_year}.")

                    # --- PLOT RESULTS ---
                    st.subheader("Visualisasi Prediksi")
                    plt.figure(figsize=(12, 6))

                    # Plot data historis
                    plt.plot(df.index, df["Produksi"], label="Data Aktual", color="blue", marker='o')

                    # Persiapkan data untuk prediksi jika ada
                    if not future_df.empty:
                        # Plot prediksi
                        plt.plot(future_df.index, future_df["Produksi"], label="Prediksi (Smoothed)", 
                                color="red", linestyle="--", marker='x')

                    plt.axvline(x=df.index.max(), color="black", linestyle=":", label="Awal Prediksi")
                    plt.title(f"Prediksi Produksi Energi {energy_type}")
                    plt.xlabel("Tahun")
                    plt.ylabel("Produksi")
                    plt.grid(True)
                    plt.legend()
                    st.pyplot(plt)

                except KeyError as e:
                    st.subheader("Hasil Prediksi")
                    st.write(f"Tidak ada data prediksi untuk Tahun {target_year}. Detail: {e}")
                except Exception as e:
                    st.subheader("Hasil Prediksi")
                    st.write(f"Terjadi error: {e}")
            else:
                st.error("Data Biodiesel tidak cukup untuk membuat prediksi setelah menerapkan lag.")
    else:
        # Untuk model SVR (termasuk Fuel Ethanol) - data bulanan
        future_df = forecast_production_svr(target_year, model, scaler, df)

        # --- DISPLAY RESULTS ---
        try:
            # Ambil hasil prediksi untuk bulan Desember di tahun target
            december_prediction = future_df.loc[f"{target_year}-12", "Produksi"]

            # Pastikan hasil prediksi adalah scalar
            if isinstance(december_prediction, pd.Series):
                december_prediction = december_prediction.iloc[0]

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
        plt.axvline(x=pd.to_datetime("2025-01-01"), color="black", linestyle=":", label="Awal Prediksi")
        plt.title(f"Prediksi Produksi Energi {energy_type}")
        plt.xlabel("Tahun")
        plt.ylabel("Produksi")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
except Exception as e:
    st.error(f"Terjadi error saat memuat model atau data: {e}")
    if energy_type == "Fuel Ethanol":
        st.info("Pastikan file model 'svr_fuel_ethanol_model.pkl' dan scaler 'scaler_fuel_ethanol.pkl' tersedia di folder 'materials'.")
    elif energy_type == "Biodiesel":
        st.info("Pastikan file model 'linreg_biodiesel_model.pkl' dan scaler 'scaler_biodiesel.pkl' tersedia di folder 'materials'.")
