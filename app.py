import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# LOAD MODEL DAN SCALER
# ================================
linreg_biodiesel = joblib.load("materials/linreg_biodiesel_model.pkl")
scaler_biodiesel = joblib.load("materials/scaler_biodiesel.pkl")

svr_coal = joblib.load("materials/svr_coal_model.pkl")
scaler_coal = joblib.load("materials/scaler_coal.pkl")

svr_fuel_ethanol = joblib.load("materials/svr_fuel_ethanol_model.pkl")
scaler_fuel_ethanol = joblib.load("materials/scaler_fuel_ethanol.pkl")

svr_natural_gas = joblib.load("materials/svr_natural_gas_model.pkl")
scaler_natural_gas = joblib.load("materials/scaler_natural_gas.pkl")

svr_petroleum = joblib.load("materials/svr_petroleum_model.pkl")
scaler_petroleum = joblib.load("materials/scaler_petroleum.pkl")

# ================================
# STREAMLIT APP
# ================================
st.set_page_config(page_title="Forecast Produksi Energi", layout="centered")
st.title("📊 Forecast Produksi Energi hingga 2050")
st.write("Masukkan tahun (2024 - 2050) untuk melihat prediksi produksi energi.")

# Input tahun
tahun_input = st.number_input("Pilih Tahun:", min_value=2024, max_value=2050, step=1)

# ================================
# GENERIC PREDICT FUNCTION
# ================================
def predict_future(model, scaler, last_value, start_year, target_year):
    """
    Recursive prediction berbasis lag_1.
    last_value : produksi terakhir (float)
    start_year : tahun terakhir di dataset (misalnya 2023)
    target_year: tahun yang diminta user
    """
    future_value = last_value
    for _ in range(target_year - start_year):
        lag_input = np.array([[future_value]])
        lag_scaled = scaler.transform(lag_input)
        future_value = model.predict(lag_scaled)[0]
    return future_value

# ================================
# NILAI PRODUKSI TERAKHIR (2023)
# ⚠️ Ganti sesuai nilai aktual dataset training kamu
# ================================
last_year_data = 2023
last_values = {
    "Biodiesel": 10.0,     # TODO: ganti dengan produksi biodiesel 2023
    "Coal": 50.0,          # TODO: ganti dengan produksi coal 2023
    "Fuel Ethanol": 8.0,   # TODO: ganti dengan produksi ethanol 2023
    "Natural Gas": 100.0,  # TODO: ganti dengan produksi gas alam 2023
    "Petroleum": 70.0      # TODO: ganti dengan produksi petroleum 2023
}

# ================================
# PREDICTION
# ================================
predictions = {}
predictions["Biodiesel"] = predict_future(linreg_biodiesel, scaler_biodiesel, last_values["Biodiesel"], last_year_data, tahun_input)
predictions["Coal"] = predict_future(svr_coal, scaler_coal, last_values["Coal"], last_year_data, tahun_input)
predictions["Fuel Ethanol"] = predict_future(svr_fuel_ethanol, scaler_fuel_ethanol, last_values["Fuel Ethanol"], last_year_data, tahun_input)
predictions["Natural Gas"] = predict_future(svr_natural_gas, scaler_natural_gas, last_values["Natural Gas"], last_year_data, tahun_input)
predictions["Petroleum"] = predict_future(svr_petroleum, scaler_petroleum, last_values["Petroleum"], last_year_data, tahun_input)

# ================================
# OUTPUT
# ================================
st.subheader(f"Hasil Prediksi Tahun {tahun_input}")
df_result = pd.DataFrame(list(predictions.items()), columns=["Jenis Energi", "Produksi (Prediksi)"])
st.table(df_result)

# Chart
st.subheader("Visualisasi Prediksi")
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_result["Jenis Energi"], df_result["Produksi (Prediksi)"],
       color=["blue", "gray", "green", "orange", "red"])
ax.set_ylabel("Produksi")
ax.set_title(f"Prediksi Produksi Energi Tahun {tahun_input}")
st.pyplot(fig)

# Footer
st.caption("⚡ Model forecasting ini berbasis SVR & Linear Regression dengan residual smoothing.")
