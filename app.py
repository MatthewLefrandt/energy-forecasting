import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- FUNGSI EVALUASI ---
def evaluate_model(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mape, mae, mse, rmse, r2

# --- SMOOTHING FUNCTION ---
def moving_average_smoothing(series, window=3):
    return pd.Series(series).rolling(window=window, center=True, min_periods=1).mean()

# --- STREAMLIT APP ---
st.title("Prediksi Produksi Biodiesel")
st.write("Aplikasi ini memprediksi produksi biodiesel berdasarkan model Linear Regression yang telah dilatih.")

# --- USER INPUT ---
st.sidebar.header("Input Prediksi")
target_year = st.sidebar.number_input("Masukkan Tahun Prediksi", min_value=2025, max_value=2050, value=2025, step=1)

# --- LOAD DATA ---
@st.cache
def load_data():
    df = pd.read_excel("materials/Combined_modelling.xlsx")
    df = df.loc[:, ~df.apply(lambda col: col.isna().all() or (col.astype(str).str.strip() == "").all())]
    df = df[df["Jenis_Energi"] == "Biodiesel"].drop(columns=["Tipe_Aliran"])
    df = df.T.reset_index()
    df.columns = ['Tahun', 'Produksi']
    df = df[1:]
    df["Tahun"] = df["Tahun"].astype(int)
    df["Produksi"] = df["Produksi"].astype(float)
    df = df.set_index("Tahun")
    return df

df = load_data()

# --- PREPROCESSING ---
df["Lag_1"] = df["Produksi"].shift(1)
df = df.dropna()

train_df = df[df.index <= 2018]
test_df = df[df.index > 2018]

X_train = train_df[["Lag_1"]].values
y_train = train_df["Produksi"].values
X_test = test_df[["Lag_1"]].values
y_test = test_df["Produksi"].values

# --- LOAD MODEL & SCALER ---
model = joblib.load("materials/linreg_biodiesel_model.pkl")
scaler = joblib.load("materials/scaler_biodiesel.pkl")

# --- PREDIKSI TRAIN/TEST ---
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# --- EVALUASI ---
mape_train, mae_train, mse_train, rmse_train, r2_train = evaluate_model(y_train, y_train_pred)
mape_test, mae_test, mse_test, rmse_test, r2_test = evaluate_model(y_test, y_test_pred)

st.subheader("Evaluasi Model")
st.write("**In-Sample (Train):**")
st.write(f"MAPE: {mape_train:.2f}%, MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, R²: {r2_train:.4f}")

st.write("**Out-of-Sample (Test):**")
st.write(f"MAPE: {mape_test:.2f}%, MAE: {mae_test:.2f}, MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, R²: {r2_test:.4f}")

# --- FORECASTING ---
last_year = df.index.max()
future_years = list(range(last_year + 1, target_year + 1))
recent_value = df["Produksi"].values[-1]
future_preds = []

for _ in future_years:
    lag_input = np.array([[recent_value]])
    lag_scaled = scaler.transform(lag_input)
    pred = model.predict(lag_scaled)[0]
    future_preds.append(pred)
    recent_value = pred  # update untuk tahun berikutnya

# --- SMOOTHING ---
y_train_pred_smoothed = moving_average_smoothing(y_train_pred, window=3)
y_test_pred_smoothed = moving_average_smoothing(y_test_pred, window=3)
future_preds_smoothed = moving_average_smoothing(future_preds, window=3)

# --- VISUALISASI ---
st.subheader("Visualisasi Prediksi")
plt.figure(figsize=(12, 6))

# Plot Train
plt.plot(train_df.index, y_train, label="Train Actual", color="blue")
plt.plot(train_df.index, y_train_pred_smoothed, label="Train Predicted (Smoothed)", color="blue", linestyle="--")

# Plot Test
plt.plot(test_df.index, y_test, label="Test Actual", color="orange")
plt.plot(test_df.index, y_test_pred_smoothed, label="Test Predicted (Smoothed)", color="orange", linestyle="--")

# Plot Forecast
plt.plot(future_years, future_preds_smoothed, label=f"Forecast (Smoothed, {last_year + 1}–{target_year})", color="red", linestyle="--", marker="x")

# Add vertical lines
plt.axvline(x=2018, color="gray", linestyle=":", label="Test Start")
plt.axvline(x=last_year, color="black", linestyle=":", label="Forecast Start")

# Add titles and labels
plt.title("Prediksi Produksi Biodiesel Tahunan (Linear Regression + Moving Average Smoothing)")
plt.xlabel("Tahun")
plt.ylabel("Produksi")
plt.grid(True)
plt.legend()
st.pyplot(plt)

# --- DISPLAY FORECAST RESULTS ---
st.subheader("Hasil Prediksi")
st.write(f"Prediksi Produksi Biodiesel untuk Tahun {target_year}: {future_preds_smoothed.iloc[-1]:.2f}")
