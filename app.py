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

# --- ENERGY RESERVES --- 
ENERGY_RESERVES = {
    "Batu Bara": 21_444_852.31,  # Dalam trilliun BTU
    "Gas Alam": 7_172_147.19,  # Dalam trilliun BTU
    "Minyak Bumi": 9_390_178.86,  # Dalam trilliun BTU
    "Biodiesel": 0,  # Placeholder
    "Fuel Ethanol": 0  # Placeholder
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

# --- RESERVE CALCULATION FUNCTION ---
def calculate_remaining_reserves(energy_type, historical_df, future_df, target_year):
    """
    Menghitung cadangan energi yang tersisa dan estimasi tahun habisnya.

    Args:
        energy_type (str): Jenis energi yang dipilih
        historical_df (DataFrame): Data historis produksi
        future_df (DataFrame): Data prediksi produksi
        target_year (int): Tahun prediksi target

    Returns:
        tuple: (cadangan_tersisa, estimasi_tahun_habis, persentase_tersisa)
    """
    if energy_type not in ENERGY_RESERVES or ENERGY_RESERVES[energy_type] == 0:
        return None, None, None

    try:
        # Ambil total cadangan awal
        initial_reserves = ENERGY_RESERVES[energy_type]

        # 1. Dapatkan data historis dari file Excel (1980-2023)
        raw_data = pd.read_excel(DATA_PATH)
        raw_data = raw_data[raw_data["Jenis_Energi"] == energy_type]

        # Ambil data tahunan dari 1980-2023
        historical_years = range(1980, 2024)  # dari 1980 hingga 2023
        historical_consumption = 0

        # Jika data bentuknya sudah tahunan
        if not raw_data.empty:
            # Ambil baris data untuk energi yang dipilih
            energy_data = raw_data.iloc[0]

            # Jumlahkan konsumsi historis (1980-2023)
            for year in historical_years:
                year_str = str(year)
                if year_str in energy_data and not pd.isna(energy_data[year_str]):
                    historical_consumption += float(energy_data[year_str])

        # 2. Dapatkan data prediksi dari tahun 2024 sampai tahun target
        future_consumption = 0

        if energy_type == "Biodiesel":
            # Untuk data biodiesel yang sudah dalam format tahunan
            for year in range(2024, target_year + 1):
                if year in future_df.index:
                    value = future_df.loc[year, "Produksi"]
                    if isinstance(value, pd.Series):
                        value = value.iloc[0]  # Ambil nilai pertama jika Series
                    future_consumption += value
        else:
            # Untuk data lain yang dalam format bulanan, ambil hanya bulan Desember sebagai nilai tahunan
            for year in range(2024, target_year + 1):
                december = f"{year}-12"
                if december in future_df.index:
                    value = future_df.loc[december, "Produksi"]
                    if isinstance(value, pd.Series):
                        value = value.iloc[0]  # Ambil nilai pertama jika Series
                    future_consumption += value

        # 3. Hitung cadangan tersisa
        remaining_reserves = initial_reserves - historical_consumption - future_consumption

        # 4. Hitung persentase cadangan tersisa
        percentage_remaining = (remaining_reserves / initial_reserves) * 100

        # 5. Prediksi tahun habisnya berdasarkan laju konsumsi terakhir
        if energy_type == "Biodiesel":
            # Gunakan nilai prediksi tahun terakhir
            if target_year in future_df.index:
                annual_consumption_rate = future_df.loc[target_year, "Produksi"]
                if isinstance(annual_consumption_rate, pd.Series):
                    annual_consumption_rate = annual_consumption_rate.iloc[0]
            else:
                annual_consumption_rate = future_df["Produksi"].iloc[-1]
        else:
            # Gunakan nilai prediksi Desember tahun terakhir
            december_target = f"{target_year}-12"
            if december_target in future_df.index:
                annual_consumption_rate = future_df.loc[december_target, "Produksi"]
                if isinstance(annual_consumption_rate, pd.Series):
                    annual_consumption_rate = annual_consumption_rate.iloc[0]
            else:
                # Ambil data Desember terakhir yang ada
                try:
                    # Cari bulan Desember terakhir yang tersedia
                    december_dates = [date for date in future_df.index if date.month == 12]
                    if december_dates:
                        last_december = max(december_dates)
                        annual_consumption_rate = future_df.loc[last_december, "Produksi"]
                        if isinstance(annual_consumption_rate, pd.Series):
                            annual_consumption_rate = annual_consumption_rate.iloc[0]
                    else:
                        annual_consumption_rate = future_df["Produksi"].iloc[-1]
                except:
                    annual_consumption_rate = future_df["Produksi"].iloc[-1]

        # Jika cadangan tersisa negatif, cari tahun habisnya antara tahun awal dan target
        if remaining_reserves <= 0:
            # Reset nilai konsumsi historis
            cumulative_consumption = historical_consumption

            # Cari tahun di mana cadangan habis
            for year in range(2024, target_year + 1):
                if energy_type == "Biodiesel":
                    if year in future_df.index:
                        year_consumption = future_df.loc[year, "Produksi"]
                        if isinstance(year_consumption, pd.Series):
                            year_consumption = year_consumption.iloc[0]
                    else:
                        continue
                else:
                    december = f"{year}-12"
                    if december in future_df.index:
                        year_consumption = future_df.loc[december, "Produksi"]
                        if isinstance(year_consumption, pd.Series):
                            year_consumption = year_consumption.iloc[0]
                    else:
                        continue

                cumulative_consumption += year_consumption
                if cumulative_consumption >= initial_reserves:
                    # Interpolasi untuk mendapatkan fraksi tahun yang lebih akurat
                    prev_consumption = cumulative_consumption - year_consumption
                    remaining = initial_reserves - prev_consumption
                    fraction = remaining / year_consumption
                    estimated_year = year - 1 + fraction

                    # Kembalikan nilai cadangan yang mungkin negatif jika target_year melebihi tahun habis
                    if target_year > estimated_year:
                        over_consumption = cumulative_consumption - initial_reserves
                        # Buat nilai percentage_remaining negatif untuk menunjukkan habisnya
                        percentage_remaining = -1 * (over_consumption / initial_reserves) * 100
                        return remaining_reserves, estimated_year, percentage_remaining

                    return 0, estimated_year, 0

            # Jika tidak bisa ditemukan tahun pastinya, gunakan nilai sebelum tahun target
            return remaining_reserves, target_year - 1, percentage_remaining
        else:
            # Hitung sisa tahun berdasarkan konsumsi tahunan terakhir
            years_remaining = remaining_reserves / annual_consumption_rate if annual_consumption_rate > 0 else float('inf')
            estimated_year_depleted = target_year + years_remaining

        return remaining_reserves, estimated_year_depleted, percentage_remaining

    except Exception as e:
        print(f"Error dalam calculate_remaining_reserves: {e}")
        return None, None, None

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

                        # Tambahkan metrik cadangan tersisa untuk batu bara, gas alam, dan minyak bumi
                        if energy_type in ["Batu Bara", "Gas Alam", "Minyak Bumi"]:
                            st.markdown("### Estimasi Cadangan")

                            # Hitung cadangan tersisa
                            remaining_reserves, depletion_year, percentage_remaining = calculate_remaining_reserves(
                                energy_type, df, future_df, target_year
                            )

                            if remaining_reserves is not None:
                                metrics_col1, metrics_col2 = st.columns(2)
                                # Bagian metrik cadangan tersisa
                                with metrics_col1:
                                    # Warna metrik berdasarkan sisa cadangan
                                    # "normal" -> hijau jika positif
                                    # "inverse" -> merah jika negatif
                                    delta_color = "normal" if percentage_remaining > 0 else "inverse"
                                
                                    display_value = f"{abs(remaining_reserves):,.2f}"
                                    if remaining_reserves < 0:
                                        display_value = f"-{display_value}"
                                
                                    st.metric(
                                        "Cadangan Tersisa", 
                                        display_value, 
                                        f"{percentage_remaining:.1f}% dari total",
                                        delta_color=delta_color
                                    )

            except (KeyError, Exception) as e:
                st.warning(f"Tidak ada hasil prediksi untuk Desember {target_year}.")

        # Visualisasi
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
                ),
                plot_bgcolor='rgba(0,0,0,0)',  # Background transparan
                # Tambahkan konfigurasi warna teks dan grid horizontal
                xaxis=dict(
                    showgrid=False,
                    gridcolor='rgba(0,0,0,0)',
                    tickfont=dict(color='white'),
                    title_font=dict(color='white')
                ),
                yaxis=dict(
                    showgrid=False,
                    gridcolor='rgba(0,0,0,0)',
                    tickfont=dict(color='white'),
                    title_font=dict(color='white')
                ),
                font=dict(color='white'),
                title_font=dict(color='white')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Tampilkan visualisasi cadangan tersisa jika energi fosil
            if energy_type in ["Batu Bara", "Gas Alam", "Minyak Bumi"]:
                # Hitung cadangan tersisa
                remaining_reserves, depletion_year, percentage_remaining = calculate_remaining_reserves(
                    energy_type, df, future_df, target_year
                )

                if remaining_reserves is not None:
                    st.markdown("### Visualisasi Cadangan Tersisa")

                    # Warna berdasarkan persentase tersisa
                    if percentage_remaining <= 0:
                        gauge_color = "red"
                        display_value = 0  # Minimum untuk gauge
                    elif percentage_remaining < 20:
                        gauge_color = "red"
                        display_value = percentage_remaining
                    elif percentage_remaining < 50:
                        gauge_color = "orange"
                        display_value = percentage_remaining
                    else:
                        gauge_color = ENERGY_COLORS.get(energy_type, '#1E88E5')
                        display_value = percentage_remaining

                    # Warna delta pada gauge selalu merah untuk persentase negatif
                    delta_properties = {
                        'reference': 100, 
                        'decreasing': {'color': "red"}, 
                        'suffix': '%',
                        'font': {'size': 16}
                    }
                    if percentage_remaining < 0:
                        delta_properties['valueformat'] = '.1f'  # Format untuk nilai negatif
                        delta_properties['decreasing']['color'] = "red"

                    # Buat speedometer chart dengan Plotly yang lebih menarik dan interaktif
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=max(0, display_value),  # Pastikan tidak negatif untuk gauge
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={
                            'text': f"<b>Cadangan {energy_type} Tersisa</b>", 
                            'font': {'size': 24, 'family': 'Arial, sans-serif', 'color': 'white'}  # Ubah ke putih
                        },
                        delta=delta_properties,
                        number={
                            'suffix': '%',
                            'font': {'size': 26, 'family': 'Arial, sans-serif', 'color': gauge_color}
                        },
                        gauge={
                            'axis': {
                                'range': [0, 100], 
                                'tickwidth': 1, 
                                'tickcolor': "rgba(0,0,0,0)",  # Warna tick yang transparan
                                'tickfont': {'size': 14, 'color': 'white'},  # Ubah ke putih
                                'showticklabels': True  # Tetap tampilkan label
                            },
                            'bar': {'color': gauge_color, 'thickness': 0.7},
                            'bgcolor': 'rgba(255, 255, 255, 0)',  # Sepenuhnya transparan
                            'borderwidth': 0,  # Tanpa border
                            'bordercolor': "rgba(0,0,0,0)",  # Transparan
                            'steps': [
                                {'range': [0, 20], 'color': 'rgba(255, 99, 71, 0.25)'},  # Merah sangat transparan
                                {'range': [20, 50], 'color': 'rgba(255, 165, 0, 0.25)'},  # Oranye sangat transparan
                                {'range': [50, 100], 'color': 'rgba(144, 238, 144, 0.25)'}  # Hijau sangat transparan
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 3},
                                'thickness': 0.8,
                                'value': 20
                            }
                        }
                    ))

                    # Tambahkan teks keterangan nilai numerik (tanpa T BTU)
                    reserve_text = f"{abs(remaining_reserves):,.0f}"
                    if remaining_reserves < 0:
                        reserve_text = f"-{reserve_text} (Defisit)"
                        text_color = "red"
                    else:
                        text_color = "white"  # Ubah ke putih

                    gauge_fig.add_annotation(
                        x=0.5, y=0.3,
                        text=reserve_text,
                        font={'size': 22, 'color': text_color, 'family': 'Arial, sans-serif', 'weight': 'bold'},
                        showarrow=False,
                        hovertext="Total cadangan yang tersisa (nilai absolut)"
                    )

                    # Tambahkan teks estimasi tahun habis dengan tampilan yang lebih baik
                    if percentage_remaining <= 0:
                        depletion_text = f"Cadangan Habis Sekitar Tahun {depletion_year:.1f}!"
                        text_color = "red"
                    elif np.isinf(depletion_year):
                        depletion_text = "Laju produksi sangat rendah"
                        text_color = "green"
                    else:
                        years_until_depletion = depletion_year - datetime.now().year
                        if years_until_depletion < 50:
                            text_color = "red"
                        elif years_until_depletion < 100:
                            text_color = "orange"
                        else:
                            text_color = "green"
                        depletion_text = f"Estimasi habis pada tahun {depletion_year:.1f}"

                    gauge_fig.add_annotation(
                        x=0.5, y=0.15,
                        text=depletion_text,
                        font={'size': 16, 'color': text_color, 'family': 'Arial, sans-serif'},
                        showarrow=False,
                        hovertext="Tahun estimasi ketika cadangan akan habis"
                    )

                    # Layout yang lebih menarik dan konsisten dengan visualisasi prediksi
                    gauge_fig.update_layout(
                        height=450,
                        margin=dict(l=20, r=20, t=60, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",  # Paper background transparan
                        plot_bgcolor="rgba(0,0,0,0)",   # Plot background transparan
                        font={'color': "white", 'family': "Arial, sans-serif"},  # Ubah semua font ke putih
                        template="plotly_white",
                        hovermode="closest",
                        hoverlabel=dict(
                            bgcolor="white",
                            font_size=14,
                            font_family="Arial, sans-serif"
                        ),
                        showlegend=False
                    )

                    # Tambahkan interaktivitas melalui tooltip kustom
                    gauge_fig.add_trace(go.Scatter(
                        x=[0.5],
                        y=[0.5],
                        mode="markers",
                        marker=dict(
                            size=1,
                            color="rgba(0,0,0,0)"
                        ),
                        hoverinfo="text",
                        hovertext=f"<b>Detail Cadangan {energy_type}</b><br>" +
                                 f"Total cadangan awal: {ENERGY_RESERVES[energy_type]:,.0f}<br>" +
                                 f"Cadangan tersisa: {reserve_text}<br>" +
                                 f"Persentase tersisa: {max(0, percentage_remaining):.1f}%<br>" + 
                                 f"{depletion_text}",
                        showlegend=False
                    ))

                    # Tampilkan chart
                    st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})

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
st.markdown('</div>', unsafe_allow_html=True
