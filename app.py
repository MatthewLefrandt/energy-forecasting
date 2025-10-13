import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time

# --- SETUP PAGE CONFIG ---
st.set_page_config(
    page_title="Prediksi Produksi Energi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RESERVE DATA ---
RESERVE_DATA = {
    "Batu Bara": {
        "total_reserve": 21_444_852.31,  # Triliun BTU
        "total_produced_2023": 5_308_167.64  # Triliun BTU
    },
    "Gas Alam": {
        "total_reserve": 7_172_147.19,  # Triliun BTU
        "total_produced_2023": 4_358_683.78  # Triliun BTU
    },
    "Minyak Bumi": {
        "total_reserve": 9_390_178.86,  # Triliun BTU
        "total_produced_2023": 7_192_856.88  # Triliun BTU
    }
}

# --- WELCOME POPUP USING SESSION STATE ---
if 'popup_shown' not in st.session_state:
    st.session_state.popup_shown = False

# Function to close popup
def close_popup():
    st.session_state.popup_shown = True
    st.rerun()

# --- CONDITIONAL DISPLAY BASED ON POPUP STATE ---
if not st.session_state.popup_shown:
    # Show only the welcome message when popup is active - without the white box background
    st.markdown("<div style='text-align: center; padding: 40px; margin-bottom: 30px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 60px; margin-bottom: 20px;'>👋</div>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: #1E88E5; font-size: 36px; margin-bottom: 30px;'>Selamat Datang di Aplikasi Prediksi Produksi Energi!</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size: 22px; margin-bottom: 40px; line-height: 1.6;'>
    Untuk pengalaman terbaik, kami menyarankan Anda:
    <br><br>
    <b>1.</b> Gunakan <b>Light Mode</b> pada browser Anda
    <br>
    <b>2.</b> Atur <b>Zoom Browser</b> ke <b>75%</b> untuk tampilan optimal
    <br><br>
    Penyesuaian ini akan mencegah elemen tampilan saling tumpang tindih dan memastikan visualisasi data terlihat dengan sempurna.
    </p>
    """, unsafe_allow_html=True)

    # Center the button and make it larger
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <style>
        div.stButton > button {
            font-size: 20px;
            font-weight: bold;
            padding: 12px 24px;
            background-color: #1E88E5;
            color: white;
            border: none;
            border-radius: 8px;
        }
        div.stButton > button:hover {
            background-color: #1565C0;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("Mengerti, Lanjutkan", key="popup_button", use_container_width=True):
            close_popup()

    st.markdown("</div>", unsafe_allow_html=True)

    # Stop rendering rest of the app
    st.stop()

# --- CUSTOM CSS --- (only shown after popup is closed)
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
    .prediction-section {
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .prediction-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 15px;
    }
    .prediction-value {
        font-size: 4rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 20px 0 25px 0; /* Added more padding at bottom */
        background: linear-gradient(to right, rgba(30, 136, 229, 0.1), rgba(30, 136, 229, 0.2), rgba(30, 136, 229, 0.1));
        border-radius: 10px;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(30, 136, 229, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(30, 136, 229, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(30, 136, 229, 0);
        }
    }
    .reserve-container {
        text-align: center;
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        background: rgba(248, 249, 250, 0.8);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .reserve-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .reserve-label {
        font-size: 1rem;
        color: #757575;
    }
    .depletion-info {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 15px;
        padding: 8px;
        border-radius: 5px;
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

# --- HEADER --- (only once, at the top)
st.markdown('<h1 class="main-header">Prediksi Produksi Energi</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Visualisasi dan prediksi produksi energi berdasarkan model machine learning</p>', unsafe_allow_html=True)

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

# --- RESERVE CALCULATION FUNCTION ---
def calculate_reserves(energy_type, future_df, target_year):
    """Calculate remaining reserves for a given energy type up to target year."""

    if energy_type not in RESERVE_DATA:
        return None, None, None

    total_reserve = RESERVE_DATA[energy_type]["total_reserve"]
    total_produced_2023 = RESERVE_DATA[energy_type]["total_produced_2023"]

    # Initial remaining reserve at end of 2023
    remaining_reserve_2023 = total_reserve - total_produced_2023

    # For yearly data visualization, aggregate monthly predictions to yearly sums
    if not future_df.empty and isinstance(future_df.index, pd.DatetimeIndex):
        yearly_future = future_df.resample('Y').sum().reset_index()
        yearly_future["year"] = yearly_future["Tahun"].dt.year
        yearly_future = yearly_future.set_index("year")
    else:
        yearly_future = future_df.copy() if not future_df.empty else pd.DataFrame()

    # Create a range of years to analyze
    start_year = 2024
    end_year = max(target_year, 2100)  # Extend to at least 2100 to find depletion year

    # Initialize dictionary to track reserves over time
    reserves_by_year = {2023: remaining_reserve_2023}

    # Initial values
    current_reserve = remaining_reserve_2023
    depletion_year = None

    # Calculate reserves for each future year
    for year in range(start_year, end_year + 1):
        if year in yearly_future.index:
            yearly_production = yearly_future.loc[year, "Produksi"]
            if isinstance(yearly_production, pd.Series):
                yearly_production = yearly_production.iloc[0]
        else:
            # If we don't have a prediction, use the last available year's production
            # and apply a simple growth rate (you can adjust this assumption)
            if not yearly_future.empty:
                last_year_prod = yearly_future["Produksi"].iloc[-1]
                if isinstance(last_year_prod, pd.Series):
                    last_year_prod = last_year_prod.iloc[0]
                yearly_production = last_year_prod * 1.01  # Assuming 1% annual growth
            else:
                # Fallback if no predictions available
                yearly_production = 0

        # Update reserves
        current_reserve -= yearly_production
        reserves_by_year[year] = current_reserve

        # Check for depletion
        if depletion_year is None and current_reserve <= 0:
            depletion_year = year

    # Create dataframe from the reserves data
    reserves_df = pd.DataFrame({
        'Year': list(reserves_by_year.keys()),
        'Remaining_Reserve': list(reserves_by_year.values())
    })

    # Get remaining reserve at target year
    target_reserve = reserves_by_year.get(target_year, 0)

    return reserves_df, target_reserve, depletion_year

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

# --- FORMAT FUNCTION ---
def format_dataframe(df):
    """Format numeric columns with commas for display."""
    df_display = df.copy()
    for col in df_display.select_dtypes(include=['float64', 'int64']).columns:
        df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}")
    return df_display

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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Parameter Prediksi")

    energy_type = st.selectbox(
        "Jenis Energi",
        options=list(MODEL_PATHS.keys()),
        format_func=lambda x: f"{ENERGY_ICONS.get(x, '⚡')} {x}"
    )

    min_year = 2025
    max_year = 2100  # Diubah dari 2050 menjadi 2100
    default_year = 2030

    target_year = st.slider(
        "Tahun Prediksi", 
        min_value=min_year, 
        max_value=max_year, 
        value=default_year,
        step=1,
        help="Pilih tahun untuk memprediksi produksi energi hingga 2100"
    )

    # Tambahkan peringatan untuk prediksi jangka sangat panjang
    if target_year > 2050:
        st.sidebar.warning(f"Prediksi hingga tahun {target_year} merupakan ekstrapolasi jangka sangat panjang. Hasilnya mungkin kurang akurat.")

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
                    st.markdown(f"<div class='prediction-section'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='prediction-title'>Hasil Prediksi {target_year}</div>", unsafe_allow_html=True)

                    if target_year in future_df.index:
                        year_prediction = future_df.loc[target_year, "Produksi"]
                        if isinstance(year_prediction, pd.Series):
                            year_prediction = year_prediction.iloc[0]

                        if not pd.isnull(year_prediction):
                            # Format with commas for thousands and add unit inside the blue box
                            formatted_prediction = f"{year_prediction:,.2f}"
                            st.markdown(f"""
                            <div class='prediction-value'>
                                {formatted_prediction}
                                <div style='font-size: 1.2rem; margin-top: -10px;'>Trilliun BTU</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Data ringkasan
                            st.markdown("### Ringkasan Data")
                            last_actual = df["Produksi"].iloc[-1]
                            growth = ((year_prediction - last_actual) / last_actual) * 100

                            metrics_col1, metrics_col2 = st.columns(2)
                            with metrics_col1:
                                st.metric(
                                    "Data Terakhir", 
                                    f"{last_actual:,.2f}", 
                                    f"Tahun {df.index[-1]}"
                                )
                            with metrics_col2:
                                st.metric(
                                    "Pertumbuhan", 
                                    f"{growth:.2f}%", 
                                    f"Dari {df.index[-1]} ke {target_year}",
                                    delta_color="normal" if growth >= 0 else "inverse"
                                )

                    st.markdown("</div>", unsafe_allow_html=True)

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

                    # Layout with updated y-axis title
                    fig.update_layout(
                        title=f"Produksi {energy_type} (Historis dan Prediksi)",
                        xaxis_title="Tahun",
                        yaxis_title="Produksi (Trilliun BTU)",  # Added unit
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

                    # Tambahkan tabel ringkasan dengan format angka dengan koma
                    with st.expander("Tabel Data Prediksi", expanded=False):
                        future_table = future_df.reset_index()
                        future_table.columns = ["Tahun", "Produksi"]
                        # Filter only years from 2025 onward
                        future_table = future_table[future_table["Tahun"] >= 2025]
                        st.dataframe(format_dataframe(future_table), use_container_width=True)
    else:
        # Prediksi untuk model SVR (data bulanan)
        future_df = forecast_production_svr(target_year, model, scaler, df)

        # Tampilkan hasil
        with col2:
            st.markdown(f"<div class='prediction-section'>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-title'>Hasil Prediksi {target_year}</div>", unsafe_allow_html=True)

            try:
                december_prediction = future_df.loc[f"{target_year}-12", "Produksi"]
                if isinstance(december_prediction, pd.Series):
                    december_prediction = december_prediction.iloc[0]

                if not pd.isnull(december_prediction):
                    # Format with commas for thousands and add unit inside the blue box
                    formatted_prediction = f"{december_prediction:,.2f}"
                    st.markdown(f"""
                    <div class='prediction-value'>
                        {formatted_prediction}
                        <div style='font-size: 1.2rem; margin-top: -10px;'>Trilliun BTU</div>
                    </div>
                    """, unsafe_allow_html=True)

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
                                f"{last_actual:,.2f}", 
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

            st.markdown("</div>", unsafe_allow_html=True)

            # Tambahkan visualisasi cadangan energi untuk energi fosil
            if energy_type in ["Batu Bara", "Gas Alam", "Minyak Bumi"]:
                # Hitung cadangan
                reserves_df, target_reserve, depletion_year = calculate_reserves(energy_type, future_df, target_year)

                if reserves_df is not None:
                    st.markdown("### Status Cadangan Energi")

                    # Format remaining reserves for display
                    if target_reserve <= 0:
                        reserve_color = "red"
                        reserve_text = "HABIS 🚨"
                        reserve_value = "0"
                    else:
                        reserve_color = "#1E88E5" if target_reserve > RESERVE_DATA[energy_type]["total_reserve"] * 0.3 else "#FFA726" if target_reserve > RESERVE_DATA[energy_type]["total_reserve"] * 0.1 else "red"
                        reserve_text = "TERSISA"
                        reserve_value = f"{target_reserve:,.2f} Trilliun BTU"

                    # Display reserve status
                    st.markdown(f"""
                    <div class='reserve-container' style='border: 2px solid {reserve_color};'>
                        <div class='reserve-label'>Status Cadangan {energy_type} di Tahun {target_year}</div>
                        <div class='reserve-value' style='color: {reserve_color};'>{reserve_text}</div>
                        <div style='font-size: 1.5rem; margin-bottom: 10px;'>{reserve_value}</div>

                        <div class='depletion-info' style='background-color: {"rgba(255,82,82,0.1)" if depletion_year else "rgba(76,175,80,0.1)"}; color: {"#D32F2F" if depletion_year else "#2E7D32"};'>
                            {f"Diperkirakan habis pada tahun {depletion_year}" if depletion_year and depletion_year <= 2100 else "Cadangan masih tersedia hingga setelah tahun 2100"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

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

            # Layout with updated y-axis title
            fig.update_layout(
                title=f"Produksi {energy_type} (Historis dan Prediksi)",
                xaxis_title="Tahun",
                yaxis_title="Produksi (Trilliun BTU)",  # Added unit
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

            # Tambah tabel ringkasan dengan format angka dengan koma
            with st.expander("Tabel Data Prediksi", expanded=False):
                yearly_future = future_df.resample('Y').mean().reset_index()
                yearly_future["Tahun"] = yearly_future["Tahun"].dt.year
                yearly_future = yearly_future.rename(columns={"Tahun": "Tahun", "Produksi": "Produksi (rata-rata)"})
                # Filter only years from 2025 onward
                yearly_future = yearly_future[yearly_future["Tahun"] >= 2025]
                st.dataframe(format_dataframe(yearly_future), use_container_width=True)

            # Tambahkan visualisasi cadangan energi untuk energi fosil
            if energy_type in ["Batu Bara", "Gas Alam", "Minyak Bumi"]:
                reserves_df, target_reserve, depletion_year = calculate_reserves(energy_type, future_df, target_year)

                if reserves_df is not None:
                    st.markdown("### Visualisasi Cadangan Energi")

                    # Visualisasi grafik cadangan
                    fig_reserves = go.Figure()

                    # Add trace for remaining reserves
                    fig_reserves.add_trace(go.Scatter(
                        x=reserves_df['Year'],
                        y=reserves_df['Remaining_Reserve'],
                        mode='lines',
                        name='Cadangan Tersisa',
                        line=dict(color=ENERGY_COLORS.get(energy_type, '#1E88E5'), width=3),
                        fill='tozeroy'  # Fill area under the line
                    ))

                    # Add horizontal line at zero
                    fig_reserves.add_shape(
                        type="line",
                        x0=min(reserves_df['Year']),
                        y0=0,
                        x1=max(reserves_df['Year']),
                        y1=0,
                        line=dict(color="red", width=2, dash="dash"),
                    )

                    # Add vertical line at depletion year if it exists
                    if depletion_year:
                        fig_reserves.add_vline(
                            x=depletion_year,
                            line_width=2,
                            line_dash="dash",
                            line_color="red",
                        )

                        # Add annotation for depletion year
                        fig_reserves.add_annotation(
                            x=depletion_year,
                            y=0,
                            text=f"Habis Tahun {depletion_year}",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40,
                            font=dict(color="red", size=14)
                        )

                    # Add vertical line at target year
                    fig_reserves.add_vline(
                        x=target_year,
                        line_width=2,
                        line_dash="dash",
                        line_color="#1E88E5",
                    )

                    # Add annotation for target year
                    target_reserve_value = reserves_df.loc[reserves_df['Year'] == target_year, 'Remaining_Reserve'].values[0]
                    fig_reserves.add_annotation(
                        x=target_year,
                        y=target_reserve_value,
                        text=f"Tahun {target_year}",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        font=dict(color="#1E88E5", size=14)
                    )

                    # Update layout
                    fig_reserves.update_layout(
                        title=f"Proyeksi Cadangan {energy_type} (2023-{max(reserves_df['Year'])})",
                        xaxis_title="Tahun",
                        yaxis_title="Cadangan Tersisa (Trilliun BTU)",
                        template="plotly_white",
                        height=500,
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_reserves, use_container_width=True)

                    # Tambahkan visualisasi tabung cadangan energi
                    st.markdown("### Visualisasi Tabung Cadangan")

                    # Calculate initial and remaining capacity
                    initial_reserve = RESERVE_DATA[energy_type]["total_reserve"] - RESERVE_DATA[energy_type]["total_produced_2023"]
                    percent_remaining = (target_reserve / initial_reserve) * 100 if initial_reserve > 0 else 0
                    percent_remaining = max(0, min(100, percent_remaining))

                    # Set color based on remaining percentage
                    if percent_remaining <= 0:
                        gauge_color = "red"
                        status_text = "HABIS"
                    elif percent_remaining < 20:
                        gauge_color = "red"
                        status_text = "KRITIS"
                    elif percent_remaining < 50:
                        gauge_color = "orange"
                        status_text = "MENIPIS"
                    else:
                        gauge_color = "green"
                        status_text = "AMAN"

                    # Create a gauge chart as a tank visualization
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=percent_remaining,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Cadangan {energy_type}<br><span style='font-size:0.8em;color:gray'>Tahun {target_year}</span>", 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': gauge_color},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 20], 'color': 'rgba(255, 0, 0, 0.3)'},
                                {'range': [20, 50], 'color': 'rgba(255, 165, 0, 0.3)'},
                                {'range': [50, 100], 'color': 'rgba(0, 128, 0, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 20
                            }
                        },
                        number={'suffix': "%", 'font': {'size': 24}},
                        delta={'reference': 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}}
                    ))

                    # Add text annotation for status
                    fig_gauge.add_annotation(
                        x=0.5,
                        y=0.25,
                        text=f"Status: {status_text}",
                        font=dict(size=18, color=gauge_color),
                        showarrow=False
                    )

                    # Add additional info below gauge
                    if target_reserve <= 0:
                        info_text = f"Cadangan habis pada tahun {depletion_year if depletion_year else '?'}"
                    else:
                        remaining_years = depletion_year - target_year if depletion_year else ">77"
                        info_text = f"Perkiraan bertahan: {remaining_years} tahun lagi"

                    fig_gauge.add_annotation(
                        x=0.5,
                        y=0.15,
                        text=info_text,
                        font=dict(size=14),
                        showarrow=False
                    )

                    # Adjust layout
                    fig_gauge.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=60, b=20)
                    )

                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Tambahkan tabel data cadangan
                    with st.expander("Tabel Data Cadangan", expanded=False):
                        # Extract relevant years for table display
                        key_years = [2023]
                        display_years = list(range(2025, target_year + 1, 5))
                        if target_year not in display_years:
                            display_years.append(target_year)
                        if depletion_year and depletion_year not in display_years and depletion_year <= 2100:
                            display_years.append(depletion_year)
                        display_years = sorted(key_years + display_years)

                        # Filter DataFrame for these years
                        table_df = reserves_df[reserves_df['Year'].isin(display_years)].copy()
                        table_df.columns = ['Tahun', 'Cadangan Tersisa (Trilliun BTU)']

                        # Add status column
                        def get_status(reserve):
                            if reserve <= 0:
                                return "HABIS 🚨"
                            elif reserve < initial_reserve * 0.2:
                                return "KRITIS ⚠️"
                            elif reserve < initial_reserve * 0.5:
                                return "MENIPIS ⚠️"
                            else:
                                return "AMAN ✅"

                        table_df['Status'] = table_df['Cadangan Tersisa (Trilliun BTU)'].apply(get_status)

                        # Display formatted table
                        st.dataframe(format_dataframe(table_df), use_container_width=True)

except FileNotFoundError:
    st.error(f"File model atau data tidak ditemukan untuk energi {energy_type}.")
    st.info("Pastikan semua file model dan data tersedia di folder 'materials'.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
    st.info("Coba pilih jenis energi lain atau sesuaikan parameter prediksi.")

# --- FOOTER ---
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('© Matthew Lefrandt 2025 Prediksi Produksi Energi | Dibuat dengan Streamlit', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
