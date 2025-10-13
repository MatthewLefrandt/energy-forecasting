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

# --- CALCULATE REMAINING RESERVES FUNCTION ---
def calculate_remaining_reserves(energy_type, future_df, target_year):
    """Calculate remaining reserves based on current and future production."""

    # Initial reserves and production data
    initial_reserves = {
        "Batu Bara": 21_444_852.31,
        "Gas Alam": 7_172_147.19,
        "Minyak Bumi": 9_390_178.86
    }

    produced_until_2023 = {
        "Batu Bara": 5_308_167.64,
        "Gas Alam": 4_358_683.78,
        "Minyak Bumi": 7_192_856.88
    }

    if energy_type not in initial_reserves:
        return None, None, None  # Return None for non-fossil fuels

    # Calculate remaining reserves as of end of 2023
    remaining_2023 = initial_reserves[energy_type] - produced_until_2023[energy_type]

    # Get yearly production from the forecast
    if not future_df.empty:
        # Create a yearly production Series from the forecasted data
        if energy_type != "Biodiesel":  # For monthly data models
            # For monthly data, aggregate to get annual totals
            yearly_production = future_df.resample('Y').sum()
        else:  # For yearly data models
            # Already yearly, just use as is
            yearly_production = future_df

        # Initialize reserves calculation starting from remaining_2023
        yearly_reserves = {}
        current_reserve = remaining_2023

        # Start with 2023 value for visualization
        yearly_reserves[2023] = current_reserve

        # Calculate reserves for each forecasted year by subtracting that year's production
        for idx, row in yearly_production.iterrows():
            year = idx.year if hasattr(idx, 'year') else idx  # Handle both datetime and int indices

            if year >= 2024:  # Only process years from 2024 onwards
                production = row['Produksi']
                current_reserve -= production
                yearly_reserves[year] = current_reserve

        # Create a series with all calculated reserves
        years = sorted(yearly_reserves.keys())
        reserves_series = pd.Series([yearly_reserves[y] for y in years], index=years)

        # Find depletion year (first negative value)
        negative_years = [y for y in years if yearly_reserves[y] <= 0]
        depletion_year = None if not negative_years else negative_years[0]

        # Get reserves at target year
        if target_year in yearly_reserves:
            reserves_at_target = yearly_reserves[target_year]
        elif target_year > max(years):
            # If target year is beyond our calculation, estimate using average production rate
            last_year = max(years)
            last_reserve = yearly_reserves[last_year]
            years_beyond = target_year - last_year

            # Calculate average yearly production from the last few years
            recent_years = [y for y in years if y >= 2024][-5:]  # Last 5 years or fewer
            if len(recent_years) >= 2:
                avg_yearly_production = sum(yearly_production.loc[str(y)]['Produksi'] 
                                          if energy_type != "Biodiesel" else yearly_production.loc[y]['Produksi'] 
                                          for y in recent_years) / len(recent_years)
            else:
                # If not enough data, use the only available year
                avg_yearly_production = yearly_production.iloc[-1]['Produksi']

            reserves_at_target = last_reserve - (avg_yearly_production * years_beyond)
        else:
            # Target year is before our first calculation (shouldn't happen)
            reserves_at_target = remaining_2023

        return reserves_series, depletion_year, reserves_at_target

    # If no future data, return just the 2023 value
    return pd.Series([remaining_2023], index=[2023]), None, remaining_2023

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

        # Add reserves projection for fossil fuels
        if energy_type in ["Batu Bara", "Gas Alam", "Minyak Bumi"]:
            # Calculate reserves
            remaining_reserves, depletion_year, reserves_at_target = calculate_remaining_reserves(
                energy_type, future_df, target_year)

            if remaining_reserves is not None:
                # Create an expander for the reserves visualization
                with st.expander("Proyeksi Cadangan Energi", expanded=True):
                    st.markdown(f"### Proyeksi Cadangan {energy_type}")

                    reserves_col1, reserves_col2 = st.columns([2, 1])

                    with reserves_col1:
                        # Create a visualization of reserves over time
                        fig_reserves = go.Figure()

                        # Convert index to years for cleaner display
                        years_index = [date.year for date in remaining_reserves.index]

                        # Add trace for remaining reserves
                        fig_reserves.add_trace(go.Scatter(
                            x=years_index,
                            y=remaining_reserves.values,
                            fill='tozeroy',
                            mode='lines',
                            name='Sisa Cadangan',
                            line=dict(color=ENERGY_COLORS.get(energy_type, '#1E88E5'), width=2),
                            fillcolor=f"rgba({','.join(map(str, [int(int(ENERGY_COLORS.get(energy_type, '#1E88E5')[1:3], 16)), int(ENERGY_COLORS.get(energy_type, '#1E88E5')[3:5], 16), int(ENERGY_COLORS.get(energy_type, '#1E88E5')[5:7], 16)]))},.3)"
                        ))

                        # Add depletion line if applicable
                        if depletion_year:
                            fig_reserves.add_vline(
                                x=depletion_year, 
                                line_width=2, 
                                line_dash="dash", 
                                line_color="red"
                            )

                            fig_reserves.add_annotation(
                                x=depletion_year,
                                y=remaining_reserves.max() * 0.9,
                                text=f"Cadangan Habis ({depletion_year})",
                                showarrow=True,
                                arrowhead=1,
                                ax=40,
                                ay=-40,
                                font=dict(color="red", size=14)
                            )

                        # Layout
                        fig_reserves.update_layout(
                            title=f"Proyeksi Cadangan {energy_type} Hingga Habis",
                            xaxis_title="Tahun",
                            yaxis_title="Sisa Cadangan (Trilliun BTU)",
                            template="plotly_white",
                            height=400,
                            hovermode="x unified"
                        )

                        st.plotly_chart(fig_reserves, use_container_width=True)

                    with reserves_col2:
                        # Display cylinder visualization for remaining reserves
                        percentage_remaining = 0
                        initial_reserves = {
                            "Batu Bara": 21_444_852.31,
                            "Gas Alam": 7_172_147.19,
                            "Minyak Bumi": 9_390_178.86
                        }

                        if reserves_at_target is not None:
                            percentage_remaining = max(0, min(100, (reserves_at_target / initial_reserves[energy_type]) * 100))

                        # Create cylinder chart
                        fig_cylinder = go.Figure()

                        # Draw cylinder 
                        fig_cylinder.add_trace(go.Scatter(
                            x=[0, 0, 1, 1, 0],
                            y=[0, 1, 1, 0, 0],
                            mode='lines',
                            line=dict(color='black', width=2),
                            showlegend=False
                        ))

                        # Fill cylinder based on remaining percentage
                        y_fill = percentage_remaining / 100

                        # For completely empty, still show a thin line
                        if y_fill <= 0.01:
                            y_fill = 0.01

                        fig_cylinder.add_trace(go.Scatter(
                            x=[0, 0, 1, 1, 0],
                            y=[0, y_fill, y_fill, 0, 0],
                            fill="toself",
                            fillcolor=ENERGY_COLORS.get(energy_type, '#1E88E5'),
                            line=dict(color=ENERGY_COLORS.get(energy_type, '#1E88E5')),
                            showlegend=False
                        ))

                        # Add percentage text in the middle
                        fig_cylinder.add_annotation(
                            x=0.5,
                            y=y_fill/2 if y_fill > 0.15 else y_fill + 0.15,
                            text=f"{percentage_remaining:.1f}%",
                            showarrow=False,
                            font=dict(
                                size=20,
                                color='white' if y_fill > 0.15 else 'black'
                            )
                        )

                        # Layout
                        fig_cylinder.update_layout(
                            title=f"Sisa Cadangan<br>Tahun {target_year}",
                            template="plotly_white",
                            height=300,
                            xaxis=dict(
                                showticklabels=False,
                                showgrid=False,
                                zeroline=False,
                                range=[-0.1, 1.1]
                            ),
                            yaxis=dict(
                                showticklabels=False,
                                showgrid=False,
                                zeroline=False,
                                range=[-0.1, 1.1]
                            ),
                            margin=dict(l=10, r=10, t=70, b=10),
                            plot_bgcolor='rgba(0,0,0,0)'
                        )

                        st.plotly_chart(fig_cylinder, use_container_width=True)

                        # Display numerical information
                        if reserves_at_target is not None:
                            reserves_status = "Tersedia" if reserves_at_target > 0 else "Habis"
                            reserves_color = "green" if reserves_at_target > 0 else "red"

                            st.markdown(f"""
                            <div style='text-align: center; margin-top: 10px;'>
                                <div style='font-size: 16px; margin-bottom: 5px;'>Status Cadangan:</div>
                                <div style='font-size: 24px; font-weight: bold; color: {reserves_color};'>{reserves_status}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown(f"""
                            <div style='text-align: center; margin-top: 15px;'>
                                <div style='font-size: 16px; margin-bottom: 5px;'>Sisa Cadangan:</div>
                                <div style='font-size: 24px; font-weight: bold;'>{reserves_at_target:,.2f} Trilliun BTU</div>
                            </div>
                            """, unsafe_allow_html=True)

                            if depletion_year:
                                years_until_empty = max(0, depletion_year - datetime.now().year)
                                st.markdown(f"""
                                <div style='text-align: center; margin-top: 15px;'>
                                    <div style='font-size: 16px; margin-bottom: 5px;'>Habis Pada Tahun:</div>
                                    <div style='font-size: 24px; font-weight: bold;'>{depletion_year}</div>
                                    <div style='font-size: 14px; color: gray;'>({years_until_empty} tahun lagi)</div>
                                </div>
                                """, unsafe_allow_html=True)

                        # Add disclaimer
                        st.markdown("""
                        <div style='margin-top: 30px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 12px;'>
                            <i>* Proyeksi berdasarkan data produksi saat ini dan asumsi tidak ada penemuan cadangan baru.</i>
                        </div>
                        """, unsafe_allow_html=True)

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
