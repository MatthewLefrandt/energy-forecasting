import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- SETUP PAGE CONFIG ---
st.set_page_config(
    page_title="Prediksi Produksi Energi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INISIALISASI STATE ---
if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True

# --- TAMPILKAN HALAMAN BERDASARKAN STATE ---
if st.session_state.show_welcome:
    # CSS untuk halaman welcome
    st.markdown("""
    <style>
        .welcome-page {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 80vh;
            text-align: center;
            padding: 0 20px;
        }
        .welcome-card {
            background-color: white;
            border-radius: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }
        .welcome-icon {
            font-size: 72px;
            margin-bottom: 24px;
        }
        .welcome-title {
            font-size: 28px;
            color: #1E88E5;
            margin-bottom: 24px;
            font-weight: bold;
        }
        .welcome-text {
            font-size: 18px;
            color: #424242;
            line-height: 1.6;
            margin-bottom: 32px;
        }
        .stButton button {
            background-color: #1E88E5 !important;
            color: white !important;
            font-size: 18px !important;
            padding: 12px 24px !important;
            border-radius: 50px !important;
            font-weight: bold !important;
            border: none !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            background-color: #1565C0 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
            transform: translateY(-2px) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Tampilkan welcome page sebagai halaman terpisah
    st.markdown("""
    <div class="welcome-page">
        <div class="welcome-card">
            <div class="welcome-icon">🔍✨</div>
            <div class="welcome-title">Selamat Datang di Aplikasi Prediksi Energi</div>
            <div class="welcome-text">
                Di sini Anda bisa memprediksi jumlah produksi berbagai jenis energi hingga tahun 2100, melihat visualisasi interaktif tren historis dan masa depan, serta menganalisis keberlanjutan cadangan energi fosil dan potensi penggantiannya dengan energi terbarukan.
                Untuk pengalaman terbaik dalam menjelajahi visualisasi data kami, 
                kami merekomendasikan untuk:<br><br>
                • Mengatur zoom browser ke <b>70%</b><br>
                • Menggunakan mode terang (Light Mode)<br><br>
                Pengaturan ini akan membantu Anda melihat detail grafik dan 
                perbandingan data dengan lebih optimal.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tombol continue yang ditempatkan di tengah
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Mengerti, Lanjutkan", key="continue_btn", use_container_width=True):
            st.session_state.show_welcome = False
            st.rerun()

else:
    # CSS untuk aplikasi utama
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
    DIESEL_DATA_PATH = "materials/df_produksi_distilatefueloil_for_deployment.xlsx"
    GASOLINE_DATA_PATH = "materials/df_produksi_motorgasoline_for_deployment.xlsx"

    # --- ENERGY RESERVES --- 
    ENERGY_RESERVES = {
        "Batu Bara": 21_444_852.31,
        "Gas Alam": 7_172_147.19,
        "Minyak Bumi": 9_390_178.86,
        "Biodiesel": 0,
        "Fuel Ethanol": 0
    }

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

    @st.cache_data
    def load_replacement_data():
        """Load data for diesel and gasoline production/prediction."""
        diesel_df = pd.read_excel(DIESEL_DATA_PATH)
        diesel_df.columns = ['Tahun', 'Produksi_Prediksi']
        diesel_df = diesel_df[diesel_df['Tahun'] <= 2100]

        gasoline_df = pd.read_excel(GASOLINE_DATA_PATH)
        gasoline_df.columns = ['Tahun', 'Produksi_Prediksi']
        gasoline_df = gasoline_df[gasoline_df['Tahun'] <= 2100]

        return diesel_df, gasoline_df

    # --- SMOOTHING FUNCTION ---
    def moving_average_smoothing(series, window=6):
        """Fungsi untuk melakukan smoothing menggunakan moving average."""
        return pd.Series(series).rolling(window=window, center=True, min_periods=1).mean()

    # --- FORECAST FUNCTIONS ---
    def forecast_production_svr(year, model, scaler, data, energy_type):
        """Prediksi dengan model SVR (data bulanan) dan berhenti ketika energi habis."""
        last_date = data.index[-1]
        target_date = pd.to_datetime(f"{year}-12-01")
        future_months = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), end=target_date, freq="MS")

        recent_values = data["Produksi"].values[-12:].tolist()
        future_preds = []

        # Untuk energi tak terbarukan, hitung kapan habisnya
        depletion_year = None
        if energy_type in ["Batu Bara", "Gas Alam", "Minyak Bumi"]:
            initial_reserves = ENERGY_RESERVES[energy_type]

            # Hitung konsumsi historis
            raw_data = pd.read_excel(DATA_PATH)
            raw_data = raw_data[raw_data["Jenis_Energi"] == energy_type]
            historical_consumption = 0

            if not raw_data.empty:
                energy_data = raw_data.iloc[0]
                for year_hist in range(1980, last_date.year + 1):
                    year_str = str(year_hist)
                    if year_str in energy_data and not pd.isna(energy_data[year_str]):
                        historical_consumption += float(energy_data[year_str])

            remaining = initial_reserves - historical_consumption

            # Prediksi hingga energi habis atau tahun target tercapai
            cumulative_future = 0

            for month_idx, future_month in enumerate(future_months):
                if remaining <= 0:
                    depletion_year = future_months[max(0, month_idx-1)].year
                    break

                lag_input = np.array(recent_values[-12:]).reshape(1, -1)
                lag_scaled = scaler.transform(lag_input)
                pred = model.predict(lag_scaled)[0]

                # Kurangi cadangan tersisa
                if future_month.month == 12:  # Hanya hitung konsumsi tahunan di bulan Desember
                    cumulative_future += pred
                    remaining = initial_reserves - historical_consumption - cumulative_future

                future_preds.append(pred)
                recent_values.append(pred)

            # Jika energi habis sebelum target_date, potong future_months
            if depletion_year is not None:
                # Ambil bulan terakhir dari tahun habisnya
                last_valid_date = pd.Timestamp(year=depletion_year, month=12, day=1)
                future_months = [date for date in future_months if date <= last_valid_date]
        else:
            # Untuk energi terbarukan, prediksi seperti biasa
            for _ in range(len(future_months)):
                lag_input = np.array(recent_values[-12:]).reshape(1, -1)
                lag_scaled = scaler.transform(lag_input)
                pred = model.predict(lag_scaled)[0]
                future_preds.append(pred)
                recent_values.append(pred)

        # Pastikan panjang future_preds sesuai dengan future_months
        future_preds = future_preds[:len(future_months)]

        future_df = pd.DataFrame({
            "Tahun": future_months,
            "Produksi": future_preds
        }).set_index("Tahun")

        future_df["Produksi"] = moving_average_smoothing(future_df["Produksi"])
        return future_df, depletion_year

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

    # --- RESERVE CALCULATION FUNCTION ---
    def calculate_remaining_reserves(energy_type, historical_df, future_df, target_year):
        """
        Menghitung cadangan energi yang tersisa dan estimasi tahun habisnya.
        """
        if energy_type not in ENERGY_RESERVES or ENERGY_RESERVES[energy_type] == 0:
            return None, None, None

        try:
            initial_reserves = ENERGY_RESERVES[energy_type]

            # 1. Data historis dari file Excel (1980-2023)
            raw_data = pd.read_excel(DATA_PATH)
            raw_data = raw_data[raw_data["Jenis_Energi"] == energy_type]

            historical_years = range(1980, 2024)
            historical_consumption = 0

            if not raw_data.empty:
                energy_data = raw_data.iloc[0]
                for year in historical_years:
                    year_str = str(year)
                    if year_str in energy_data and not pd.isna(energy_data[year_str]):
                        historical_consumption += float(energy_data[year_str])

            # 2. Data prediksi dari tahun 2024 sampai tahun target
            future_consumption = 0

            if energy_type == "Biodiesel":
                for year in range(2024, target_year + 1):
                    if year in future_df.index:
                        value = future_df.loc[year, "Produksi"]
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        future_consumption += value
            else:
                for year in range(2024, target_year + 1):
                    december = pd.Timestamp(year=year, month=12, day=1)
                    if december in future_df.index:
                        value = future_df.loc[december, "Produksi"]
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        future_consumption += value

            # 3. Hitung cadangan tersisa
            remaining_reserves = initial_reserves - historical_consumption - future_consumption

            # 4. Hitung persentase cadangan tersisa
            percentage_remaining = (remaining_reserves / initial_reserves) * 100

            # 5. Prediksi tahun habisnya
            if energy_type == "Biodiesel":
                if target_year in future_df.index:
                    annual_consumption_rate = future_df.loc[target_year, "Produksi"]
                    if isinstance(annual_consumption_rate, pd.Series):
                        annual_consumption_rate = annual_consumption_rate.iloc[0]
                else:
                    annual_consumption_rate = future_df["Produksi"].iloc[-1]
            else:
                december_target = pd.Timestamp(year=target_year, month=12, day=1)
                if december_target in future_df.index:
                    annual_consumption_rate = future_df.loc[december_target, "Produksi"]
                    if isinstance(annual_consumption_rate, pd.Series):
                        annual_consumption_rate = annual_consumption_rate.iloc[0]
                else:
                    december_dates = [date for date in future_df.index if date.month == 12]
                    if december_dates:
                        last_december = max(december_dates)
                        annual_consumption_rate = future_df.loc[last_december, "Produksi"]
                        if isinstance(annual_consumption_rate, pd.Series):
                            annual_consumption_rate = annual_consumption_rate.iloc[0]
                    else:
                        annual_consumption_rate = future_df["Produksi"].iloc[-1]

            # Jika cadangan tersisa negatif, cari tahun habisnya
            if remaining_reserves <= 0:
                cumulative_consumption = historical_consumption

                for year in range(2024, target_year + 1):
                    if energy_type == "Biodiesel":
                        if year in future_df.index:
                            year_consumption = future_df.loc[year, "Produksi"]
                            if isinstance(year_consumption, pd.Series):
                                year_consumption = year_consumption.iloc[0]
                        else:
                            continue
                    else:
                        december = pd.Timestamp(year=year, month=12, day=1)
                        if december in future_df.index:
                            year_consumption = future_df.loc[december, "Produksi"]
                            if isinstance(year_consumption, pd.Series):
                                year_consumption = year_consumption.iloc[0]
                        else:
                            continue

                    cumulative_consumption += year_consumption
                    if cumulative_consumption >= initial_reserves:
                        prev_consumption = cumulative_consumption - year_consumption
                        remaining = initial_reserves - prev_consumption
                        fraction = remaining / year_consumption if year_consumption > 0 else 0
                        estimated_year = year - 1 + fraction

                        # Jika tahun target sudah melewati tahun habisnya, set remaining_reserves ke 0
                        if target_year > estimated_year:
                            return 0, estimated_year, 0

                        return max(0, remaining), estimated_year, max(0, percentage_remaining)

                return 0, target_year - 1, 0

            else:
                years_remaining = remaining_reserves / annual_consumption_rate if annual_consumption_rate > 0 else float('inf')
                estimated_year_depleted = target_year + years_remaining

            return max(0, remaining_reserves), estimated_year_depleted, max(0, percentage_remaining)

        except Exception as e:
            print(f"Error dalam calculate_remaining_reserves: {e}")
            return None, None, None

    # --- REPLACEMENT PROPORTION CALCULATION ---
    def calculate_replacement_proportion(renewable_df, fossil_df, target_year, energy_type):
        """Calculate the proportion of renewable energy replacing fossil fuel."""
        try:
            fossil_value = fossil_df[fossil_df['Tahun'] == target_year]['Produksi_Prediksi'].values[0]

            if energy_type == "Biodiesel":
                if target_year in renewable_df.index:
                    renewable_value = renewable_df.loc[target_year, "Produksi"]
                    if isinstance(renewable_value, pd.Series):
                        renewable_value = renewable_value.iloc[0]
                else:
                    closest_year = min(renewable_df.index, key=lambda x: abs(x - target_year))
                    renewable_value = renewable_df.loc[closest_year, "Produksi"]
                    if isinstance(renewable_value, pd.Series):
                        renewable_value = renewable_value.iloc[0]
            else:
                target_date = pd.Timestamp(year=target_year, month=12, day=1)
                if target_date in renewable_df.index:
                    renewable_value = renewable_df.loc[target_date, "Produksi"]
                    if isinstance(renewable_value, pd.Series):
                        renewable_value = renewable_value.iloc[0]
                else:
                    december_dates = [date for date in renewable_df.index 
                                  if isinstance(date, pd.Timestamp) and date.month == 12 and date.year <= target_year]
                    if december_dates:
                        last_december = max(december_dates)
                        renewable_value = renewable_df.loc[last_december, "Produksi"]
                        if isinstance(renewable_value, pd.Series):
                            renewable_value = renewable_value.iloc[0]
                    else:
                        renewable_value = 0

            proportion_percentage = (renewable_value / fossil_value) * 100

            return proportion_percentage, renewable_value, fossil_value

        except Exception as e:
            print(f"Error in calculate_replacement_proportion: {e}")
            return 0, 0, 0

    # --- HEADER ---
    st.markdown('<h1 class="main-header">Prediksi Produksi Energi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Selamat datang di Dashboard Analitik Energi yang memberikan prediksi produksi lima jenis energi utama berdasarkan data historis dan model machine learning. Navigasikan melalui sidebar untuk memilih jenis energi dan tahun target, lalu dapatkan visualisasi interaktif beserta analisis cadangan dan potensi penggantian bahan bakar yang relevan.</p>', unsafe_allow_html=True)

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
        model = joblib.load(MODEL_PATHS[energy_type])
        scaler = joblib.load(SCALER_PATHS[energy_type])
        df = load_data(energy_type)

        diesel_df, gasoline_df = load_replacement_data()

        col1, col2 = st.columns([2, 1])

        if energy_type == "Biodiesel":
            if df.empty:
                st.error("Data tidak cukup untuk melakukan prediksi.")
            else:
                df["Lag_1"] = df["Produksi"].shift(1)
                df = df.dropna()

                if not df.empty:
                    future_df = forecast_production_biodiesel(target_year, model, scaler, df)
                    depletion_year = None  # Biodiesel tidak memiliki depletion year

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

                                proportion, renewable_value, fossil_value = calculate_replacement_proportion(
                                    future_df, diesel_df, target_year, "Biodiesel"
                                )

                                metrics_col1, metrics_col2 = st.columns(2)
                                with metrics_col1:
                                    st.metric(
                                        "Produksi Biodiesel", 
                                        f"{renewable_value:,.2f}", 
                                        f"Tahun {target_year}"
                                    )
                                with metrics_col2:
                                    st.metric(
                                        "Kebutuhan Diesel", 
                                        f"{fossil_value:,.2f}", 
                                        f"Tahun {target_year}"
                                    )

                    with col1:
                        st.markdown("### Visualisasi Prediksi")

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=list(df.index), 
                            y=df["Produksi"].tolist(),
                            mode='lines+markers',
                            name='Data Historis',
                            line=dict(color=ENERGY_COLORS.get(energy_type, '#1E88E5'), width=2),
                            marker=dict(size=6)
                        ))

                        if not future_df.empty:
                            fig.add_trace(go.Scatter(
                                x=list(future_df.index), 
                                y=future_df["Produksi"].tolist(),
                                mode='lines+markers',
                                name='Prediksi',
                                line=dict(color='#FF5252', width=2, dash='dash'),
                                marker=dict(size=6, symbol='diamond')
                            ))

                        fig.add_vline(
                            x=df.index[-1], 
                            line_width=1, 
                            line_dash="dash", 
                            line_color="gray"
                        )

                        fig.add_annotation(
                            x=df.index[-1],
                            y=df["Produksi"].max(),
                            text="Mulai Prediksi",
                            showarrow=True,
                            arrowhead=1,
                            ax=40,
                            ay=-40
                        )

                        fig.update_layout(
                            title=dict(
                                text=f"Produksi {energy_type} (Historis dan Prediksi)",
                                font=dict(color='#808080', size=20)
                            ),
                            xaxis=dict(
                                showgrid=False,
                                gridcolor='rgba(0,0,0,0)',
                                tickfont=dict(color='#808080', size=14),
                                title_font=dict(color='#808080', size=16),
                                title_text="Tahun"
                            ),
                            yaxis=dict(
                                showgrid=False,
                                gridcolor='rgba(0,0,0,0)',
                                tickfont=dict(color='#808080', size=14),
                                title_font=dict(color='#808080', size=16),
                                title_text="Produksi"
                            ),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            height=500,
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                font=dict(color="#808080")
                            ),
                            font=dict(family="Arial, sans-serif", color="#808080")
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("### Potensi Penggantian Bahan Bakar Diesel")

                        proportion, renewable_value, fossil_value = calculate_replacement_proportion(
                            future_df, diesel_df, target_year, "Biodiesel"
                        )

                        renewable_color = ENERGY_COLORS.get(energy_type, '#33691E')
                        fossil_color = "#FF6F00"

                        replacement_fig = go.Figure()

                        labels = [energy_type, "Distillate Fuel Oil"]
                        values = [renewable_value, max(0, fossil_value - renewable_value)]

                        replacement_fig.add_trace(
                            go.Pie(
                                labels=labels,
                                values=values,
                                hole=0.4,
                                marker=dict(
                                    colors=[renewable_color, fossil_color],
                                    line=dict(color='#FFF', width=1)
                                ),
                                textinfo="label+percent",
                                insidetextorientation="radial",
                                textfont=dict(size=14, color='#FFF'),
                                hoverinfo="label+value+percent",
                                textposition="inside"
                            )
                        )

                        replacement_fig.update_layout(
                            height=450,
                            margin=dict(l=20, r=20, t=50, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "#808080", 'family': "Arial, sans-serif"},
                            showlegend=False,
                            title={
                                'text': f"Perbandingan Produksi Energi ({target_year})",
                                'font': {'size': 18, 'color': '#808080'}
                            }
                        )

                        st.plotly_chart(replacement_fig, use_container_width=True)

                        st.info(f"""
                        **Interpretasi:**
                        - Biodiesel dapat memenuhi sekitar {proportion:.1f}% dari kebutuhan Distillate Fuel Oil (diesel) pada tahun {target_year}.
                        - Untuk mencapai penggantian 100%, diperlukan peningkatan produksi biodiesel sebesar {(fossil_value - renewable_value):,.2f} T BTU.
                        """)

                        with st.expander("Tabel Data Prediksi", expanded=False):
                            future_table = future_df.reset_index()
                            future_table.columns = ["Tahun", "Produksi"]
                            st.dataframe(future_table, use_container_width=True)
        else:
            # Prediksi untuk model SVR (data bulanan)
            future_df, depletion_year = forecast_production_svr(target_year, model, scaler, df, energy_type)

            with col2:
                st.markdown(f"### Hasil Prediksi {target_year}")

                try:
                    # Jika energi habis sebelum target_year, tampilkan peringatan
                    if depletion_year is not None and target_year > depletion_year:
                        st.warning(f"Cadangan {energy_type} diperkirakan habis pada tahun {depletion_year}. Prediksi dihentikan.")
                        december_prediction = future_df.loc[pd.Timestamp(year=depletion_year, month=12, day=1), "Produksi"]
                        display_year = depletion_year
                    else:
                        december_prediction = future_df.loc[pd.Timestamp(year=target_year, month=12, day=1), "Produksi"]
                        display_year = target_year

                    if isinstance(december_prediction, pd.Series):
                        december_prediction = december_prediction.iloc[0]

                    if not pd.isnull(december_prediction):
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

                        if depletion_year is not None and target_year > depletion_year:
                            st.markdown(f"<p>Produksi {energy_type} Desember {depletion_year} (Terakhir):</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p>Produksi {energy_type} Desember {target_year}:</p>", unsafe_allow_html=True)

                        st.markdown(f'<p class="prediction-value">{december_prediction:.2f}</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown("### Ringkasan Data")

                        last_year_data = df[df.index.year == df.index.year.max()]
                        if not last_year_data.empty:
                            last_actual = last_year_data["Produksi"].iloc[-1]
                            last_date = last_year_data.index[-1]

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
                                    f"Dari {last_date.strftime('%b %Y')} ke Des {display_year}",
                                    delta_color="normal" if growth >= 0 else "inverse"
                                )

                            if energy_type == "Fuel Ethanol":
                                proportion, renewable_value, fossil_value = calculate_replacement_proportion(
                                    future_df, gasoline_df, display_year, "Fuel Ethanol"
                                )

                                metrics_col1, metrics_col2 = st.columns(2)
                                with metrics_col1:
                                    st.metric(
                                        "Produksi Fuel Ethanol", 
                                        f"{renewable_value:,.2f}", 
                                        f"Tahun {display_year}"
                                    )
                                with metrics_col2:
                                    st.metric(
                                        "Kebutuhan Bensin", 
                                        f"{fossil_value:,.2f}", 
                                        f"Tahun {display_year}"
                                    )

                            elif energy_type in ["Batu Bara", "Gas Alam", "Minyak Bumi"]:
                                st.markdown("### Estimasi Cadangan")

                                remaining_reserves, depletion_year_calc, percentage_remaining = calculate_remaining_reserves(
                                    energy_type, df, future_df, display_year
                                )

                                if remaining_reserves is not None:
                                    metrics_col1, metrics_col2 = st.columns(2)
                                    with metrics_col1:
                                        delta_color = "normal" if percentage_remaining > 0 else "inverse"

                                        # Jika tahun target sudah melewati tahun habisnya, tampilkan cadangan sebagai 0
                                        if depletion_year_calc is not None and target_year > depletion_year_calc:
                                            display_value = "0.00"
                                            display_percentage = "0.0"
                                        else:
                                            display_value = f"{abs(remaining_reserves):,.2f}"
                                            display_percentage = f"{max(0, percentage_remaining):.1f}"

                                        st.metric(
                                            "Cadangan Tersisa", 
                                            display_value, 
                                            f"{display_percentage}% dari total",
                                            delta_color=delta_color
                                        )

                except (KeyError, Exception) as e:
                    st.warning(f"Tidak ada hasil prediksi untuk Desember {target_year}.")

            with col1:
                st.markdown("### Visualisasi Prediksi")

                fig = go.Figure()

                yearly_df = df.resample('Y').mean()

                fig.add_trace(go.Scatter(
                    x=yearly_df.index, 
                    y=yearly_df["Produksi"],
                    mode='lines+markers',
                    name='Data Historis (Rata-rata Tahunan)',
                    line=dict(color=ENERGY_COLORS.get(energy_type, '#1E88E5'), width=2),
                    marker=dict(size=8)
                ))

                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df["Produksi"],
                    mode='lines',
                    name='Data Historis (Bulanan)',
                    line=dict(color=ENERGY_COLORS.get(energy_type, '#1E88E5'), width=1, dash='dot'),
                    opacity=0.3,
                    visible='legendonly'
                ))

                if not future_df.empty:
                    future_yearly = future_df.resample('Y').mean()

                    fig.add_trace(go.Scatter(
                        x=future_yearly.index, 
                        y=future_yearly["Produksi"],
                        mode='lines+markers',
                        name='Prediksi (Rata-rata Tahunan)',
                        line=dict(color='#FF5252', width=2),
                        marker=dict(size=8, symbol='diamond')
                    ))

                    fig.add_trace(go.Scatter(
                        x=future_df.index, 
                        y=future_df["Produksi"],
                        mode='lines',
                        name='Prediksi (Bulanan)',
                        line=dict(color='#FF5252', width=1, dash='dot'),
                        opacity=0.3,
                        visible='legendonly'
                    ))

                    # Tampilkan prediksi untuk tahun target atau tahun habisnya (mana yang lebih dulu)
                    display_year = target_year
                    if depletion_year is not None and target_year > depletion_year:
                        display_year = depletion_year

                    target_date = pd.Timestamp(year=display_year, month=12, day=1)
                    if target_date in future_df.index:
                        fig.add_trace(go.Scatter(
                            x=[target_date],
                            y=[future_df.loc[target_date, "Produksi"]],
                            mode='markers',
                            name=f'Prediksi Des {display_year}',
                            marker=dict(
                                color='#FF5252',
                                size=12,
                                symbol='star',
                                line=dict(color='#FF5252', width=2)
                            )
                        ))

                    # Highlight titik habisnya cadangan
                    if energy_type in ["Batu Bara", "Gas Alam", "Minyak Bumi"] and depletion_year is not None:
                        depletion_date = pd.Timestamp(year=depletion_year, month=12, day=1)

                        # Tambahkan garis vertikal merah putus-putus
                        fig.add_vline(
                            x=depletion_date,
                            line_width=2,
                            line_dash="dash",
                            line_color="red"
                        )

                        # Tambahkan anotasi
                        fig.add_annotation(
                            x=depletion_date,
                            y=df["Produksi"].mean() * 1.2,
                            text=f"Cadangan Habis ({depletion_year})",
                            showarrow=True,
                            arrowhead=1,
                            ax=40,
                            ay=-40,
                            font=dict(color="red", size=11)
                        )

                    last_year = df.index.year.max()
                    first_pred_year = last_year + 1
                    first_pred_date = pd.Timestamp(year=first_pred_year, month=1, day=1)

                    fig.add_vline(
                        x=first_pred_date, 
                        line_width=1, 
                        line_dash="dash", 
                        line_color="gray"
                    )

                    fig.add_annotation(
                        x=first_pred_date,
                        y=df["Produksi"].mean() * 1.2,
                        text="Mulai Prediksi",
                        showarrow=True,
                        arrowhead=1,
                        ax=40,
                        ay=-40,
                        font=dict(color="#808080")
                    )

                    fig.update_layout(
                        title=dict(
                            text=f"Produksi {energy_type} (Historis dan Prediksi)",
                            font=dict(color='#808080', size=20)
                        ),
                        xaxis=dict(
                            showgrid=False,
                            gridcolor='rgba(0,0,0,0)',
                            tickfont=dict(color='#808080', size=14),
                            title_font=dict(color='#808080', size=16),
                            title_text="Tahun"
                        ),
                        yaxis=dict(
                            showgrid=False,
                            gridcolor='rgba(0,0,0,0)',
                            tickfont=dict(color='#808080', size=14),
                            title_font=dict(color='#808080', size=16),
                            title_text="Produksi"
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=500,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(color="#808080")
                        ),
                        font=dict(family="Arial, sans-serif", color="#808080")
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    if energy_type in ["Batu Bara", "Gas Alam", "Minyak Bumi"]:
                        # Selalu gunakan tahun filter user untuk visualisasi cadangan
                        display_year = target_year
                    
                        remaining_reserves, depletion_year_calc, percentage_remaining = calculate_remaining_reserves(
                            energy_type, df, future_df, display_year
                        )
                    
                        if remaining_reserves is not None:
                            st.markdown(f"### Visualisasi Cadangan Tersisa Tahun {display_year}")
                    
                            # Cek apakah tahun target sudah melewati tahun habisnya
                            is_depleted = depletion_year_calc is not None and target_year > depletion_year_calc
                    
                            # Jika sudah habis, buat gauge chart dengan nilai 0
                            if is_depleted:
                                # Gauge chart untuk cadangan habis (0%)
                                gauge_fig = go.Figure(go.Indicator(
                                    mode="gauge",
                                    value=0,  # Nilai tetap 0
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={
                                        'text': f"<b>Cadangan {energy_type} Tersisa Tahun {display_year}</b>", 
                                        'font': {'size': 24, 'family': 'Arial, sans-serif', 'color': '#808080'}
                                    },
                                    gauge={
                                        'axis': {
                                            'range': [0, 100], 
                                            'tickwidth': 1, 
                                            'tickcolor': "rgba(0,0,0,0)",
                                            'tickfont': {'size': 14, 'color': '#808080'},
                                            'showticklabels': True
                                        },
                                        'bar': {
                                            'color': 'red',  # Warna tetap merah
                                            'thickness': 0.7
                                        },
                                        'bgcolor': 'rgba(255, 255, 255, 0)',
                                        'borderwidth': 0,
                                        'bordercolor': "rgba(0,0,0,0)",
                                        'steps': [
                                            {'range': [0, 20], 'color': 'rgba(255, 99, 71, 0.25)'},
                                            {'range': [20, 50], 'color': 'rgba(255, 165, 0, 0.25)'},
                                            {'range': [50, 100], 'color': 'rgba(144, 238, 144, 0.25)'}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 3},
                                            'thickness': 0.8,
                                            'value': 0  # Set threshold to 0
                                        }
                                    }
                                ))
                    
                                gauge_fig.add_annotation(
                                    x=0.5, y=0.43,
                                    text="0.0%",  # Teks tetap 0.0%
                                    font={'size': 64, 'color': 'red', 'family': 'Arial, sans-serif', 'weight': 'bold'},
                                    showarrow=False,
                                    align="center"
                                )
                    
                                gauge_fig.add_annotation(
                                    x=0.5, y=0.40,
                                    text="0",  # Teks tetap 0
                                    font={'size': 22, 'color': '#808080', 'family': 'Arial, sans-serif', 'weight': 'bold'},
                                    showarrow=False,
                                    align="center"
                                )
                    
                                # Hover info untuk cadangan habis
                                hover_text = f"<b>Detail Cadangan {energy_type}</b><br>" + \
                                            f"Total cadangan awal: {ENERGY_RESERVES[energy_type]:,.0f}<br>" + \
                                            f"Cadangan tersisa: 0<br>" + \
                                            f"Persentase tersisa: 0.0%<br>" + \
                                            f"Habis pada tahun: {int(depletion_year_calc)}"
                    
                            else:
                                # Gauge chart normal untuk cadangan yang masih tersisa
                                display_percentage = max(0, percentage_remaining)
                                display_value = display_percentage
                    
                                if display_value > 50:
                                    gauge_color = '#4CAF50'
                                elif display_value > 20:
                                    gauge_color = '#FFA000'
                                else:
                                    gauge_color = 'red'
                    
                                gauge_fig = go.Figure(go.Indicator(
                                    mode="gauge",
                                    value=display_value,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={
                                        'text': f"<b>Cadangan {energy_type} Tersisa Tahun {display_year}</b>", 
                                        'font': {'size': 24, 'family': 'Arial, sans-serif', 'color': '#808080'}
                                    },
                                    gauge={
                                        'axis': {
                                            'range': [0, 100], 
                                            'tickwidth': 1, 
                                            'tickcolor': "rgba(0,0,0,0)",
                                            'tickfont': {'size': 14, 'color': '#808080'},
                                            'showticklabels': True
                                        },
                                        'bar': {
                                            'color': gauge_color, 
                                            'thickness': 0.7
                                        },
                                        'bgcolor': 'rgba(255, 255, 255, 0)',
                                        'borderwidth': 0,
                                        'bordercolor': "rgba(0,0,0,0)",
                                        'steps': [
                                            {'range': [0, 20], 'color': 'rgba(255, 99, 71, 0.25)'},
                                            {'range': [20, 50], 'color': 'rgba(255, 165, 0, 0.25)'},
                                            {'range': [50, 100], 'color': 'rgba(144, 238, 144, 0.25)'}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 3},
                                            'thickness': 0.8,
                                            'value': 20
                                        }
                                    }
                                ))
                    
                                gauge_fig.add_annotation(
                                    x=0.5, y=0.43,
                                    text=f"{display_value:.1f}%",
                                    font={'size': 64, 'color': gauge_color, 'family': 'Arial, sans-serif', 'weight': 'bold'},
                                    showarrow=False,
                                    align="center"
                                )
                    
                                gauge_fig.add_annotation(
                                    x=0.5, y=0.40,
                                    text=f"{remaining_reserves:,.0f}",
                                    font={'size': 22, 'color': '#808080', 'family': 'Arial, sans-serif', 'weight': 'bold'},
                                    showarrow=False,
                                    align="center"
                                )
                    
                                # Hover info untuk cadangan masih tersisa
                                hover_text = f"<b>Detail Cadangan {energy_type}</b><br>" + \
                                            f"Total cadangan awal: {ENERGY_RESERVES[energy_type]:,.0f}<br>" + \
                                            f"Cadangan tersisa: {remaining_reserves:,.0f}<br>" + \
                                            f"Persentase tersisa: {display_value:.1f}%<br>" + \
                                            f"Estimasi habis: {int(depletion_year_calc)}"
                    
                            # Layout yang sama untuk kedua kasus
                            gauge_fig.update_layout(
                                height=450,
                                margin=dict(l=20, r=20, t=60, b=20),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font={'color': "#808080", 'family': "Arial, sans-serif"},
                                template="plotly_white",
                                hovermode="closest",
                                hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=14,
                                    font_family="Arial, sans-serif"
                                ),
                                showlegend=False
                            )
                    
                            # Tambahkan hover info
                            gauge_fig.add_trace(go.Scatter(
                                x=[0.5],
                                y=[0.5],
                                mode="markers",
                                marker=dict(
                                    size=1,
                                    color="rgba(0,0,0,0)"
                                ),
                                hoverinfo="text",
                                hovertext=hover_text,
                                showlegend=False
                            ))
                    
                            st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})

                            # Interpretasi berdasarkan tahun target user dan status cadangan
                            if depletion_year_calc is not None and target_year > depletion_year_calc:
                                st.error(f"""
                                **Interpretasi:**
                                - Cadangan {energy_type} diperkirakan sudah habis pada tahun {int(depletion_year_calc)}, sebelum target prediksi tahun {target_year}.
                                - Pada tahun {target_year}, cadangan sudah habis selama {target_year - int(depletion_year_calc)} tahun.
                                - Diperlukan penemuan cadangan baru atau pengurangan konsumsi untuk mencegah kelangkaan.
                                """)
                            elif percentage_remaining > 0:
                                st.info(f"""
                                **Interpretasi:**
                                - Cadangan {energy_type} diperkirakan tersisa sekitar {display_value:.1f}% pada tahun {display_year}.
                                - Dengan tingkat konsumsi saat ini, cadangan diperkirakan akan habis sekitar tahun {int(depletion_year_calc)}.
                                - Total cadangan tersisa sebesar {remaining_reserves:,.0f} T BTU dari total {ENERGY_RESERVES[energy_type]:,.0f} T BTU.
                                """)
                            else:
                                st.error(f"""
                                **Interpretasi:**
                                - Cadangan {energy_type} diperkirakan habis pada tahun {int(depletion_year_calc)}.
                                - Diperlukan penemuan cadangan baru atau pengurangan konsumsi untuk mencegah kelangkaan.
                                """)

                    elif energy_type == "Fuel Ethanol":
                        st.markdown("### Potensi Penggantian Bahan Bakar Bensin")

                        # Gunakan display_year untuk konsistensi
                        display_year = target_year
                        if depletion_year is not None and target_year > depletion_year:
                            display_year = depletion_year

                        proportion, renewable_value, fossil_value = calculate_replacement_proportion(
                            future_df, gasoline_df, display_year, "Fuel Ethanol"
                        )

                        renewable_color = ENERGY_COLORS.get(energy_type, '#8D6E63')
                        fossil_color = "#FF6F00"

                        replacement_fig = go.Figure()

                        labels = ["Fuel Ethanol", "Motor Gasoline"]
                        values = [renewable_value, max(0, fossil_value - renewable_value)]

                        replacement_fig.add_trace(
                            go.Pie(
                                labels=labels,
                                values=values,
                                hole=0.4,
                                marker=dict(
                                    colors=[renewable_color, fossil_color],
                                    line=dict(color='#FFF', width=1)
                                ),
                                textinfo="label+percent",
                                insidetextorientation="radial",
                                textfont=dict(size=14, color='#FFF'),
                                hoverinfo="label+value+percent",
                                textposition="inside"
                            )
                        )

                        replacement_fig.update_layout(
                            height=450,
                            margin=dict(l=20, r=20, t=50, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "#808080", 'family': "Arial, sans-serif"},
                            showlegend=False,
                            title={
                                'text': f"Perbandingan Produksi Energi ({display_year})",
                                'font': {'size': 18, 'color': '#808080'}
                            }
                        )

                        st.plotly_chart(replacement_fig, use_container_width=True)

                        st.info(f"""
                        **Interpretasi:**
                        - Fuel Ethanol dapat memenuhi sekitar {proportion:.1f}% dari kebutuhan Motor Gasoline (bensin) pada tahun {display_year}.
                        - Untuk mencapai penggantian 100%, diperlukan peningkatan produksi Fuel Ethanol sebesar {(fossil_value - renewable_value):,.2f} T BTU.
                        """)

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
    st.markdown('© 2025 Prediksi Produksi Energi | Dibuat dengan Streamlit', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
