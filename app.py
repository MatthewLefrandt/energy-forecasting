import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- SETUP PAGE CONFIG ---
st.set_page_config(
    page_title="Prediksi Produksi Energi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        color: #9e9e9e;
        font-size: 0.8rem;
        border-top: 1px solid #e0e0e0;
    }
    .reserves-container {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(to right, rgba(30, 136, 229, 0.05), rgba(30, 136, 229, 0.1), rgba(30, 136, 229, 0.05));
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .reserves-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 15px;
    }
    .reserves-subtitle {
        font-size: 1.2rem;
        color: #757575;
        margin-bottom: 25px;
    }
    .depletion-alert {
        background-color: #FFEBEE;
        color: #B71C1C;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        margin-top: 15px;
        border-left: 5px solid #B71C1C;
    }
    .safe-alert {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        margin-top: 15px;
        border-left: 5px solid #1B5E20;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER --- (only once, at the top)
st.markdown('<h1 class="main-header">Prediksi Produksi Energi</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Visualisasi dan prediksi produksi energi berdasarkan model machine learning</p>', unsafe_allow_html=True)

# --- PATHS ---
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

# --- ENERGY RESERVES ---
ENERGY_RESERVES = {
    "Batu Bara": 21_444_852.31,  # Total coal reserve in Trillion BTU
    "Gas Alam": 7_257_000.00,    # Placeholder value in Trillion BTU
    "Minyak Bumi": 1_750_000.00  # Placeholder value in Trillion BTU
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
    max_year = 2100
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

        # Tambahkan visualisasi cadangan khusus untuk Batu Bara
        if energy_type == "Batu Bara":
            st.markdown("---")
            st.markdown('<div class="reserves-container">', unsafe_allow_html=True)
            st.markdown('<h2 class="reserves-title">Cadangan Batu Bara</h2>', unsafe_allow_html=True)
            st.markdown('<p class="reserves-subtitle">Analisis cadangan tersisa dan perkiraan tahun habisnya jika tidak ditemukan sumber baru</p>', unsafe_allow_html=True)

            # Total coal reserve in Trillion BTU
            TOTAL_COAL_RESERVE = ENERGY_RESERVES["Batu Bara"]  # 21,444,852.31 Trillion BTU

            # Calculate cumulative production until 2023 (historical data)
            historical_df = df.copy()
            yearly_historical = historical_df.resample('Y').sum()
            cumulative_production_until_2023 = yearly_historical["Produksi"].sum()

            # Calculate remaining reserve as of end of 2023
            remaining_reserve_2023 = TOTAL_COAL_RESERVE - cumulative_production_until_2023

            # Create yearly forecasts for projection
            future_yearly = future_df.resample('Y').sum()

            # Project future production and calculate depletion
            projection_years = range(2024, 2200)  # Project far into the future to find depletion year
            yearly_production = {}
            cumulative_future_production = 0
            depletion_year = None
            remaining_reserves_by_year = {2023: remaining_reserve_2023}

            # Calculate year-by-year production and depletion
            last_yearly_production = yearly_historical["Produksi"].iloc[-1]
            yearly_growth_rate = 0.02  # Default 2% annual growth

            # Try to calculate growth rate from forecasts if available
            if not future_yearly.empty and len(future_yearly) > 1:
                first_forecast = future_yearly.iloc[0]["Produksi"]
                second_forecast = future_yearly.iloc[1]["Produksi"]
                if first_forecast > 0:
                    calc_growth_rate = (second_forecast / first_forecast) - 1
                    if 0 <= calc_growth_rate <= 0.2:  # Reasonable growth range 0-20%
                        yearly_growth_rate = calc_growth_rate

            for year in projection_years:
                # If we have a prediction for this year, use it
                if year in future_yearly.index.year:
                    year_production = future_yearly[future_yearly.index.year == year]["Produksi"].sum()
                else:
                    # Otherwise project based on growth rate
                    year_production = last_yearly_production * (1 + yearly_growth_rate)
                    last_yearly_production = year_production

                yearly_production[year] = year_production
                cumulative_future_production += year_production

                # Calculate remaining reserve
                remaining_reserve = remaining_reserve_2023 - cumulative_future_production
                remaining_reserves_by_year[year] = remaining_reserve

                # Check if depleted
                if remaining_reserve <= 0 and depletion_year is None:
                    depletion_year = year
                    break

            # Calculate reserves for the target year selected by user
            target_remaining = remaining_reserves_by_year.get(target_year, 0)
            if target_year >= depletion_year and depletion_year is not None:
                target_remaining = 0

            # Create columns for visualization and info
            res_col1, res_col2 = st.columns([3, 2])

            with res_col1:
                # Create tank/cylinder visualization using Plotly
                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{"type": "indicator"}, {"type": "scatter"}]],
                    column_widths=[0.4, 0.6],
                    subplot_titles=("Cadangan Tersisa di Tahun Target", "Proyeksi Cadangan Batu Bara")
                )

                # Calculate percentage remaining for tank visualization
                percentage_remaining = max(0, min(100, (target_remaining / TOTAL_COAL_RESERVE) * 100))

                # Add tank/gauge indicator
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=target_remaining,
                        number={"suffix": " Trilliun BTU", 
                                "font": {"size": 24},
                                "valueformat": ",.0f"},  # No decimals for large numbers
                        delta={
                            "reference": TOTAL_COAL_RESERVE, 
                            "position": "bottom", 
                            "valueformat": ".1%",
                            "relative": True
                        },
                        gauge={
                            "axis": {"range": [0, TOTAL_COAL_RESERVE], 
                                    "tickwidth": 1, 
                                    "visible": False},  # Hide axis for cleaner look
                            "bar": {"color": "#1E88E5" if target_remaining > 0 else "#FF5252"},
                            "bgcolor": "white",
                            "borderwidth": 2,
                            "bordercolor": "gray",
                            "steps": [
                                {"range": [0, TOTAL_COAL_RESERVE * 0.25], "color": "#FF5252"},
                                {"range": [TOTAL_COAL_RESERVE * 0.25, TOTAL_COAL_RESERVE * 0.5], "color": "#FFC107"},
                                {"range": [TOTAL_COAL_RESERVE * 0.5, TOTAL_COAL_RESERVE], "color": "#4CAF50"}
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": target_remaining
                            }
                        },
                        title={"text": f"Tahun {target_year}", "font": {"size": 20}}
                    ),
                    row=1, col=1
                )

                # Add projection line chart
                years = sorted(list(remaining_reserves_by_year.keys()))
                reserves = [remaining_reserves_by_year[year] for year in years]

                # Cut off after depletion and filter for visualization (show only future years)
                chart_years = []
                chart_reserves = []

                for i, year in enumerate(years):
                    if year >= 2023:  # Start from the last historical year
                        if remaining_reserves_by_year[year] >= 0:
                            chart_years.append(year)
                            chart_reserves.append(remaining_reserves_by_year[year])
                        else:
                            # Add the depletion point (zero reserves)
                            chart_years.append(year)
                            chart_reserves.append(0)
                            break

                # Add line chart trace
                fig.add_trace(
                    go.Scatter(
                        x=chart_years, 
                        y=chart_reserves,
                        mode='lines+markers',
                        name='Proyeksi Cadangan',
                        line=dict(color='#1E88E5', width=3),
                        fill='tozeroy'  # Fill area below the line
                    ),
                    row=1, col=2
                )

                # Add marker for target year on chart
                if target_year in chart_years:
                    target_idx = chart_years.index(target_year)
                    target_value = chart_reserves[target_idx]

                    fig.add_trace(
                        go.Scatter(
                            x=[target_year],
                            y=[target_value],
                            mode='markers',
                            marker=dict(size=12, color='red', symbol='diamond'),
                            name=f'Tahun {target_year}'
                        ),
                        row=1, col=2
                    )

                # Add depletion point marker
                if depletion_year is not None and depletion_year in chart_years:
                    fig.add_trace(
                        go.Scatter(
                            x=[depletion_year],
                            y=[0],
                            mode='markers',
                            marker=dict(size=15, color='#FF5252', symbol='x'),
                            name=f'Habis Tahun {depletion_year}'
                        ),
                        row=1, col=2
                    )

                # Update layout
                fig.update_layout(
                    height=500,
                    template="plotly_white",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.3,
                        xanchor="center",
                        x=0.5
                    )
                )

                # Update x-axis of the chart
                fig.update_xaxes(title_text="Tahun", row=1, col=2)

                # Update y-axis of the chart with formatted numbers
                fig.update_yaxes(
                    title_text="Cadangan (Trilliun BTU)", 
                    row=1, col=2,
                    tickformat=",.0f"  # Format with commas, no decimals
                )

                st.plotly_chart(fig, use_container_width=True)

            # Information display in second column
            with res_col2:
                st.markdown("### Informasi Cadangan")

                if depletion_year is not None:
                    # Calculate years until depletion
                    years_until_depletion = depletion_year - 2023

                    if target_year >= depletion_year:
                        # Depleted in target year
                        st.markdown(f"""
                        <div class="depletion-alert">
                            <div>⚠️ Cadangan Habis</div>
                            <p style="font-size: 0.9rem; font-weight: normal; margin-top: 10px;">
                            Pada tahun {target_year}, cadangan Batu Bara sudah habis sejak tahun {depletion_year}.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Calculate deficit
                        years_since_depletion = target_year - depletion_year
                        cumulative_deficit = sum(yearly_production.get(year, 0) for year in range(depletion_year, target_year + 1))

                        st.markdown(f"""
                        <div style="margin-top: 20px; padding: 15px; background-color: #FAFAFA; border-radius: 8px;">
                            <h4 style="margin-top: 0;">Defisit Energi di {target_year}</h4>
                            <p>Sumber energi alternatif atau impor diperlukan untuk menggantikan:</p>
                            <div style="font-size: 2.5rem; font-weight: bold; color: #FF5252; text-align: center; margin: 20px 0;">
                                {cumulative_deficit:,.0f} <span style="font-size: 1rem;">Trilliun BTU</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Not yet depleted in target year
                        years_remaining = depletion_year - target_year

                        st.markdown(f"""
                        <div class="safe-alert">
                            <div>✅ Cadangan Masih Tersedia</div>
                            <p style="font-size: 0.9rem; font-weight: normal; margin-top: 10px;">
                            Pada tahun {target_year}, cadangan Batu Bara masih tersedia sekitar {target_remaining:,.0f} Trilliun BTU.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style="margin-top: 20px; padding: 15px; background-color: #FAFAFA; border-radius: 8px;">
                            <h4 style="margin-top: 0;">Proyeksi Habisnya Cadangan</h4>
                            <p>Dengan pola konsumsi saat ini, cadangan Batu Bara akan habis dalam:</p>
                            <div style="font-size: 2.5rem; font-weight: bold; color: #FFC107; text-align: center; margin: 20px 0;">
                                {years_remaining} <span style="font-size: 1rem;">tahun</span>
                            </div>
                            <p style="font-size: 0.9rem; color: #616161; margin-bottom: 0;">
                            Perkiraan habis pada tahun {depletion_year}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                # Add metrics for current year vs initial reserve
                percentage_used = ((TOTAL_COAL_RESERVE - remaining_reserve_2023) / TOTAL_COAL_RESERVE) * 100

                st.markdown("### Statistik Cadangan")
                st.markdown(f"""
                <div style="display: flex; margin-top: 10px;">
                    <div style="flex: 1; background-color: #E3F2FD; padding: 10px; border-radius: 5px; margin-right: 5px;">
                        <p style="font-size: 0.8rem; color: #1565C0; margin: 0;">Total Cadangan Awal</p>
                        <p style="font-size: 1.2rem; font-weight: bold; margin: 5px 0 0 0;">{TOTAL_COAL_RESERVE:,.0f} Trilliun BTU</p>
                    </div>
                    <div style="flex: 1; background-color: #FFF3E0; padding: 10px; border-radius: 5px; margin-left: 5px;">
                        <p style="font-size: 0.8rem; color: #E65100; margin: 0;">Terpakai Hingga 2023</p>
                        <p style="font-size: 1.2rem; font-weight: bold; margin: 5px 0 0 0;">{percentage_used:.1f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Add expandable section with data table
                with st.expander("Tabel Proyeksi Cadangan", expanded=False):
                    # Create data for table, only showing a selection of years
                    projection_table = []

                    # Add key years to the table
                    display_years = [2023]  # Start with last historical year

                    # Add the target year
                    if target_year > 2023 and target_year not in display_years:
                        display_years.append(target_year)

                    # Add depletion year
                    if depletion_year is not None and depletion_year not in display_years:
                        display_years.append(depletion_year)

                    # Add intermediate years for context
                    all_years = sorted(list(remaining_reserves_by_year.keys()))
                    for year in all_years:
                        if year > 2023 and (year % 10 == 0 or year == 2025):  # Show every decade and 2025
                            if year not in display_years:
                                display_years.append(year)

                    # Sort all years to display
                    display_years.sort()

                    # Build the table
                    for year in display_years:
                        if year in remaining_reserves_by_year:
                            reserve = remaining_reserves_by_year[year]
                            # Handle negative reserves
                            reserve_display = max(0, reserve)

                            # Get production for this year if available
                            prod = yearly_production.get(year, 0) if year >= 2024 else yearly_historical["Produksi"].loc[pd.to_datetime(str(year))].sum() if year in yearly_historical.index.year else 0

                            status = "Tersedia"
                            if year >= depletion_year and depletion_year is not None:
                                status = "Habis"

                            projection_table.append({
                                "Tahun": year,
                                "Produksi": prod,
                                "Cadangan Tersisa": reserve_display,
                                "Status": status
                            })

                    # Convert to DataFrame for display
                    if projection_table:
                        proj_df = pd.DataFrame(projection_table)
                        # Format numbers with commas
                        proj_df["Produksi"] = proj_df["Produksi"].apply(lambda x: f"{x:,.2f}")
                        proj_df["Cadangan Tersisa"] = proj_df["Cadangan Tersisa"].apply(lambda x: f"{x:,.2f}")
                        st.dataframe(proj_df, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)  # Close reserves container

        # Tampilkan hasil prediksi
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

        # Visualisasi produksi energi
        with col1:
            st.markdown("### Visualisasi Prediksi Produksi")

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

except FileNotFoundError:
    st.error(f"File model atau data tidak ditemukan untuk energi {energy_type}.")
    st.info("Pastikan semua file model dan data tersedia di folder 'materials'.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
    st.info("Coba pilih jenis energi lain atau sesuaikan parameter prediksi.")

    # Add more detailed error information in an expander for debugging
    with st.expander("Detail Error"):
        st.write(e)
        st.write(f"Error Type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())

# --- FOOTER ---
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('© Matthew Lefrandt 2025 Prediksi Produksi Energi | Dibuat dengan Streamlit', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
