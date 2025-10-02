import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- SETUP PAGE CONFIG ---
st.set_page_config(
    page_title="Prediksi Produksi Energi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f5f7f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        color: #9e9e9e;
        font-size: 0.8rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- PATHS ---
DATA_PATH = "materials/Combined_modelling.xlsx"
MODEL_PATHS = {
    "Batu Bara": "materials/svr_coal_model.pkl",
    "Gas Alam": "materials/svr_natural_gas_model.pkl",
    "Minyak Bumi": "materials/svr_petroleum_model.pkl",
    "Biodiesel": "materials/linreg_biodiesel_model.pkl",
    "Fuel Ethanol": "materials/svr_fuel_ethanol_model.pkl"
}
SCALER_PATHS = {
    "Batu Bara": "materials/scaler_coal.pkl",
    "Gas Alam": "materials/scaler_natural_gas.pkl",
    "Minyak Bumi": "materials/scaler_petroleum.pkl",
    "Biodiesel": "materials/scaler_biodiesel.pkl",
    "Fuel Ethanol": "materials/scaler_fuel_ethanol.pkl"
}

# --- LOAD DATA ---
@st.cache_data
def load_data(energy_type):
    """Load dan filter data berdasarkan jenis energi."""
    df = pd.read_excel(DATA_PATH)
    df = df[df["Jenis_Energi"] == energy_type].drop(columns=["Tipe_Aliran"])
    df = df.T.reset_index()
    df.columns = ['Tahun', 'Produksi']
    df = df[1:]

    if energy_type == "Biodiesel":
        df["Tahun"] = df["Tahun"].astype(int)
        df["Produksi"] = df["Produksi"].astype(float)
        df = df.dropna()
        df = df[df["Produksi"] > 0]
        df = df.set_index("Tahun")
    else:
        df["Tahun"] = pd.to_datetime(df["Tahun"].astype(int), format='%Y')
        df["Produksi"] = df["Produksi"].astype(float)
        df = df.set_index("Tahun")
        df = df.resample("MS").interpolate(method="linear")
    return df

# --- SMOOTHING FUNCTION ---
def moving_average_smoothing(series, window=6):
    """Fungsi untuk melakukan smoothing menggunakan moving average."""
    return pd.Series(series).rolling(window=window, center=True, min_periods=1).mean()

# --- FORECAST FUNCTIONS ---
def forecast_production_svr(year, model, scaler, data):
    """Prediksi dengan model SVR (data bulanan)."""
    last_date = data.index[-1]
    target_date = pd.to_datetime(f"{year}-12-01")
    future_months = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), end=target_date, freq="MS")

    recent_values = data["Produksi"].values[-12:].tolist()
    future_preds = []

    for _ in range(len(future_months)):
        lag_input = np.array(recent_values[-12:]).reshape(1, -1)
        lag_scaled = scaler.transform(lag_input)
        pred = model.predict(lag_scaled)[0]
        future_preds.append(pred)
        recent_values.append(pred)

    future_df = pd.DataFrame({
        "Tahun": future_months,
        "Produksi": future_preds
    }).set_index("Tahun")

    future_df["Produksi"] = moving_average_smoothing(future_df["Produksi"])
    return future_df

def forecast_production_biodiesel(year, model, scaler, data):
    """Prediksi dengan model Linear Regression (data tahunan)."""
    last_year = data.index[-1]
    last_value = data["Produksi"].values[-1]

    future_years = list(range(last_year + 1, year + 1))
    if not future_years:
        return pd.DataFrame()

    future_preds = []
    recent_value = last_value

    for _ in future_years:
        lag_input = np.array([[recent_value]])
        lag_scaled = scaler.transform(lag_input)
        pred = model.predict(lag_scaled)[0]
        future_preds.append(pred)
        recent_value = pred

    future_df = pd.DataFrame({
        "Tahun": future_years,
        "Produksi": future_preds
    }).set_index("Tahun")

    future_df["Produksi"] = moving_average_smoothing(future_df["Produksi"], window=3)
    return future_df

# --- ENERGY ICONS AND COLORS ---
ENERGY_ICONS = {
    "Batu Bara": "🪨",
    "Gas Alam": "💨",
    "Minyak Bumi": "🛢️",
    "Biodiesel": "🌱",
    "Fuel Ethanol": "🌽"
}

ENERGY_COLORS = {
    "Batu Bara": "#37474F",
    "Gas Alam": "#039BE5",
    "Minyak Bumi": "#FF6F00",
    "Biodiesel": "#33691E",
    "Fuel Ethanol": "#8D6E63"
}

# --- HEADER ---
st.markdown('<h1 class="main-header">Prediksi Produksi Energi</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Visualisasi dan prediksi produksi energi berdasarkan model machine learning</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Parameter Prediksi")

    energy_type = st.selectbox(
        "Jenis Energi",
        options=list(MODEL_PATHS.keys()),
        format_func=lambda x: f"{ENERGY_ICONS.get(x, '⚡')} {x}"
    )

    min_year = 2025
    max_year = 2100
    default_year = 2026

    target_year = st.slider(
        "Tahun Prediksi", 
        min_value=min_year, 
        max_value=max_year, 
        value=default_year,
        step=1,
        help="Pilih tahun untuk memprediksi produksi energi"
    )

    st.markdown("---")
    st.markdown(f"### {ENERGY_ICONS.get(energy_type, '⚡')} {energy_type}")

    energy_descriptions = {
        "Batu Bara": "Sumber energi fosil padat yang terbentuk dari sisa tumbuhan yang terkompresi.",
        "Gas Alam": "Sumber energi fosil berbentuk gas yang terdiri dari campuran hidrokarbon.",
        "Minyak Bumi": "Sumber energi fosil cair yang terbentuk dari sisa-sisa organisme laut.",
        "Biodiesel": "Bahan bakar terbarukan yang dibuat dari minyak nabati atau lemak hewan.",
        "Fuel Ethanol": "Bahan bakar alkohol yang diproduksi dari fermentasi tanaman."
    }

    st.info(energy_descriptions.get(energy_type, ""))

# --- MAIN CONTENT ---
try:
    # Load model dan data
    model = joblib.load(MODEL_PATHS[energy_type])
    scaler = joblib.load(SCALER_PATHS[energy_type])
    df = load_data(energy_type)

    # Layout columns
    col1, col2 = st.columns([2, 1])

    # Jalankan prediksi
    if energy_type == "Biodiesel":
        if df.empty:
            st.error("Data tidak cukup untuk melakukan prediksi.")
        else:
            df["Lag_1"] = df["Produksi"].shift(1)
            df = df.dropna()

            if not df.empty:
                future_df = forecast_production_biodiesel(target_year, model, scaler, df)

                # Tampilkan hasil
                with col2:
                    st.markdown(f"### Hasil Prediksi {target_year}")

                    if target_year in future_df.index:
                        year_prediction = future_df.loc[target_year, "Produksi"]
                        if isinstance(year_prediction, pd.Series):
                            year_prediction = year_prediction.iloc[0]

                        if not pd.isnull(year_prediction):
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            st.markdown(f"<p>Produksi {energy_type} Tahun {target_year}:</p>", unsafe_allow_html=True)
                            st.markdown(f'<p class="prediction-value">{year_prediction:.2f}</p>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Data ringkasan
                            st.markdown("### Ringkasan Data")
                            last_actual = df["Produksi"].iloc[-1]
                            growth = ((year_prediction - last_actual) / last_actual) * 100

                            metrics_col1, metrics_col2 = st.columns(2)
                            with metrics_col1:
                                st.metric(
                                    "Data Terakhir", 
                                    f"{last_actual:.2f}", 
                                    f"Tahun {df.index[-1]}"
                                )
                            with metrics_col2:
                                st.metric(
                                    "Pertumbuhan", 
                                    f"{growth:.2f}%", 
                                    f"Dari {df.index[-1]} ke {target_year}",
                                    delta_color="normal" if growth >= 0 else "inverse"
                                )

                # Visualisasi
                with col1:
                    st.markdown("### Visualisasi Prediksi")

                    # Gunakan Plotly untuk visualisasi interaktif
                    fig = go.Figure()

                    # Data historis
                    fig.add_trace(go.Scatter(
                        x=list(df.index), 
                        y=df["Produksi"].tolist(),
                        mode='lines+markers',
                        name='Data Historis',
                        line=dict(color=ENERGY_COLORS.get(energy_type, '#1E88E5'), width=2),
                        marker=dict(size=6)
                    ))

                    # Data prediksi
                    if not future_df.empty:
                        fig.add_trace(go.Scatter(
                            x=list(future_df.index), 
                            y=future_df["Produksi"].tolist(),
                            mode='lines+markers',
                            name='Prediksi',
                            line=dict(color='#FF5252', width=2, dash='dash'),
                            marker=dict(size=6, symbol='diamond')
                        ))

                    # Garis vertikal pemisah (perbaikan untuk issue tanggal)
                    fig.add_vline(
                        x=df.index[-1], 
                        line_width=1, 
                        line_dash="dash", 
                        line_color="gray"
                    )

                    # Tambahkan anotasi terpisah
                    fig.add_annotation(
                        x=df.index[-1],
                        y=df["Produksi"].max(),
                        text="Mulai Prediksi",
                        showarrow=True,
                        arrowhead=1,
                        ax=40,
                        ay=-40
                    )

                    # Layout
                    fig.update_layout(
                        title=f"Produksi {energy_type} (Historis dan Prediksi)",
                        xaxis_title="Tahun",
                        yaxis_title="Produksi",
                        template="plotly_white",
                        height=500,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Tambahkan tabel ringkasan
                    with st.expander("Tabel Data Prediksi", expanded=False):
                        future_table = future_df.reset_index()
                        future_table.columns = ["Tahun", "Produksi"]
                        st.dataframe(future_table, use_container_width=True)
    else:
        # Prediksi untuk model SVR (data bulanan)
        future_df = forecast_production_svr(target_year, model, scaler, df)

        # Tampilkan hasil
        with col2:
            st.markdown(f"### Hasil Prediksi {target_year}")

            try:
                december_prediction = future_df.loc[f"{target_year}-12", "Produksi"]
                if isinstance(december_prediction, pd.Series):
                    december_prediction = december_prediction.iloc[0]

                if not pd.isnull(december_prediction):
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown(f"<p>Produksi {energy_type} Desember {target_year}:</p>", unsafe_allow_html=True)
                    st.markdown(f'<p class="prediction-value">{december_prediction:.2f}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Data ringkasan
                    st.markdown("### Ringkasan Data")

                    # Ambil data terakhir (Desember dari tahun terakhir data)
                    last_year_data = df[df.index.year == df.index.year.max()]
                    if not last_year_data.empty:
                        last_actual = last_year_data["Produksi"].iloc[-1]
                        last_date = last_year_data.index[-1]

                        # Hitung pertumbuhan
                        growth = ((december_prediction - last_actual) / last_actual) * 100

                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            st.metric(
                                "Data Terakhir", 
                                f"{last_actual:.2f}", 
                                f"{last_date.strftime('%b %Y')}"
                            )
                        with metrics_col2:
                            st.metric(
                                "Pertumbuhan", 
                                f"{growth:.2f}%", 
                                f"Dari {last_date.strftime('%b %Y')} ke Des {target_year}",
                                delta_color="normal" if growth >= 0 else "inverse"
                            )
            except (KeyError, Exception) as e:
                st.warning(f"Tidak ada hasil prediksi untuk Desember {target_year}.")

        # Visualisasi
        with col1:
            st.markdown("### Visualisasi Prediksi")

            # Gunakan Plotly untuk visualisasi interaktif
            fig = go.Figure()

            # Agregasi data bulanan ke format tahunan untuk visualisasi yang lebih bersih
            yearly_df = df.resample('Y').mean()

            # Data historis (dengan resampling tahunan untuk visualisasi)
            fig.add_trace(go.Scatter(
                x=yearly_df.index, 
                y=yearly_df["Produksi"],
                mode='lines+markers',
                name='Data Historis (Rata-rata Tahunan)',
                line=dict(color=ENERGY_COLORS.get(energy_type, '#1E88E5'), width=2),
                marker=dict(size=8)
            ))

            # Data mentah (optional, bisa dimatikan dengan parameter visible=False)
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df["Produksi"],
                mode='lines',
                name='Data Historis (Bulanan)',
                line=dict(color=ENERGY_COLORS.get(energy_type, '#1E88E5'), width=1, dash='dot'),
                opacity=0.3,
                visible='legendonly'  # Sembunyikan secara default, bisa ditampilkan dari legend
            ))

            # Agregasi data prediksi ke tahunan
            if not future_df.empty:
                future_yearly = future_df.resample('Y').mean()

                # Data prediksi (tahunan)
                fig.add_trace(go.Scatter(
                    x=future_yearly.index, 
                    y=future_yearly["Produksi"],
                    mode='lines+markers',
                    name='Prediksi (Rata-rata Tahunan)',
                    line=dict(color='#FF5252', width=2),
                    marker=dict(size=8, symbol='diamond')
                ))

                # Data prediksi (bulanan, optional)
                fig.add_trace(go.Scatter(
                    x=future_df.index, 
                    y=future_df["Produksi"],
                    mode='lines',
                    name='Prediksi (Bulanan)',
                    line=dict(color='#FF5252', width=1, dash='dot'),
                    opacity=0.3,
                    visible='legendonly'  # Sembunyikan secara default
                ))

                # Highlight khusus untuk bulan Desember tahun target
                target_date = pd.to_datetime(f"{target_year}-12-01")
                if target_date in future_df.index:
                    fig.add_trace(go.Scatter(
                        x=[target_date],
                        y=[future_df.loc[target_date, "Produksi"]],
                        mode='markers',
                        name=f'Prediksi Des {target_year}',
                        marker=dict(
                            color='#FF5252',
                            size=12,
                            symbol='star',
                            line=dict(color='#FF5252', width=2)
                        )
                    ))

            # Garis vertikal pemisah dengan perbaikan
            # Gunakan pendekatan alternatif untuk menghindari error tanggal
            last_year = df.index.year.max()
            first_pred_year = last_year + 1
            first_pred_date = f"{first_pred_year}-01-01"

            fig.add_vline(
                x=pd.to_datetime(first_pred_date), 
                line_width=1, 
                line_dash="dash", 
                line_color="gray"
            )

            # Tambahkan anotasi terpisah
            fig.add_annotation(
                x=pd.to_datetime(first_pred_date),
                y=df["Produksi"].mean() * 1.2,  # Posisikan di atas rata-rata
                text="Mulai Prediksi",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-40
            )

            # Layout
            fig.update_layout(
                title=f"Produksi {energy_type} (Historis dan Prediksi)",
                xaxis_title="Tahun",
                yaxis_title="Produksi",
                template="plotly_white",
                height=500,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Tambah tabel ringkasan (opsional, bisa dihide secara default)
            with st.expander("Tabel Data Prediksi", expanded=False):
                yearly_future = future_df.resample('Y').mean().reset_index()
                yearly_future["Tahun"] = yearly_future["Tahun"].dt.year
                yearly_future = yearly_future.rename(columns={"Tahun": "Tahun", "Produksi": "Produksi (rata-rata)"})
                st.dataframe(yearly_future, use_container_width=True)

except FileNotFoundError:
    st.error(f"File model atau data tidak ditemukan untuk energi {energy_type}.")
    st.info("Pastikan semua file model dan data tersedia di folder 'materials'.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
    st.info("Coba pilih jenis energi lain atau sesuaikan parameter prediksi.")

# --- FOOTER ---
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('© 2023 Prediksi Produksi Energi | Dibuat dengan Streamlit', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
