import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Energy Forecasting App",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .energy-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        padding: 1rem;
        background-color: #e8f5e8;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_historical_data():
    """Load historical data from Excel file"""
    try:
        df = pd.read_excel('materials/Combined_modelling.xlsx')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load all trained models and scalers"""
    models = {}
    scalers = {}
    
    # Model files mapping
    model_files = {
        'biodiesel': 'materials/linreg_biodiesel_model.pkl',
        'coal': 'materials/svr_coal_model.pkl',
        'fuel_ethanol': 'materials/svr_fuel_ethanol_model.pkl',
        'natural_gas': 'materials/svr_natural_gas_model.pkl',
        'petroleum': 'materials/svr_petroleum_model.pkl'
    }
    
    # Scaler files mapping
    scaler_files = {
        'biodiesel': 'materials/scaler_biodiesel.pkl',
        'coal': 'materials/scaler_coal.pkl',
        'fuel_ethanol': 'materials/scaler_fuel_ethanol.pkl',
        'natural_gas': 'materials/scaler_natural_gas.pkl',
        'petroleum': 'materials/scaler_petroleum.pkl'
    }
    
    # Load models
    for energy_type, file_path in model_files.items():
        try:
            with open(file_path, 'rb') as f:
                models[energy_type] = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading {energy_type} model: {str(e)}")
    
    # Load scalers
    for energy_type, file_path in scaler_files.items():
        try:
            with open(file_path, 'rb') as f:
                scalers[energy_type] = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading {energy_type} scaler: {str(e)}")
    
    return models, scalers

def create_monthly_data(yearly_data, start_year=1980):
    """Convert yearly data to monthly data"""
    monthly_data = []
    years = list(range(start_year, start_year + len(yearly_data)))
    
    for i, year_value in enumerate(yearly_data):
        if pd.notna(year_value):
            # Create 12 monthly values for each year
            for month in range(1, 13):
                monthly_data.append({
                    'year': years[i],
                    'month': month,
                    'value': year_value / 12  # Distribute yearly value across 12 months
                })
    
    return pd.DataFrame(monthly_data)

def get_lagged_features(data, energy_type, target_year, target_month):
    """Get lagged features based on energy type"""
    if energy_type == 'biodiesel':
        # Biodiesel uses 1 lag (annual data)
        lag_periods = [1]
    else:
        # Other energy types use 12 lags (monthly data)
        lag_periods = list(range(1, 13))
    
    features = []
    
    # Convert target to datetime for easier calculation
    target_date = pd.Timestamp(year=target_year, month=target_month, day=1)
    
    for lag in lag_periods:
        if energy_type == 'biodiesel':
            # For biodiesel, lag by years
            lag_date = target_date - pd.DateOffset(years=lag)
        else:
            # For others, lag by months
            lag_date = target_date - pd.DateOffset(months=lag)
        
        # Find the corresponding value in historical data
        matching_data = data[
            (data['year'] == lag_date.year) & 
            (data['month'] == lag_date.month)
        ]
        
        if not matching_data.empty:
            features.append(matching_data['value'].iloc[0])
        else:
            # If no data found, use interpolation or last known value
            prev_data = data[
                (data['year'] < lag_date.year) | 
                ((data['year'] == lag_date.year) & (data['month'] < lag_date.month))
            ]
            if not prev_data.empty:
                features.append(prev_data['value'].iloc[-1])
            else:
                features.append(0)  # Default value if no historical data
    
    return np.array(features).reshape(1, -1)

def make_prediction(energy_type, year, month, models, scalers, historical_data):
    """Make prediction for specified energy type, year, and month"""
    try:
        # Energy type mapping for data filtering
        energy_mapping = {
            'coal': 'Batu Bara',
            'natural_gas': 'Gas Alam',
            'petroleum': 'Minyak Bumi',
            'biodiesel': 'Biodiesel',
            'fuel_ethanol': 'Fuel Ethanol'
        }
        
        # Get historical data for the specific energy type
        energy_data = historical_data[historical_data['Jenis_Energi'] == energy_mapping[energy_type]]
        
        if energy_data.empty:
            return None, "No historical data found for this energy type"
        
        # Get production data (assuming 'Produksi' type)
        prod_data = energy_data[energy_data['Tipe_Aliran'] == 'Produksi']
        
        if prod_data.empty:
            return None, "No production data found"
        
        # Extract yearly values
        year_columns = [col for col in prod_data.columns if str(col).isdigit()]
        yearly_values = []
        
        for col in year_columns:
            value = prod_data[col].iloc[0]
            yearly_values.append(value if pd.notna(value) else 0)
        
        # Create monthly data if needed
        if energy_type != 'biodiesel':
            monthly_df = create_monthly_data(yearly_values, start_year=int(year_columns[0]))
        else:
            # For biodiesel, keep as yearly data but convert to DataFrame format
            years = [int(col) for col in year_columns]
            monthly_df = pd.DataFrame({
                'year': years,
                'month': [1] * len(years),  # Use month 1 for yearly data
                'value': yearly_values
            })
        
        # Get lagged features
        features = get_lagged_features(monthly_df, energy_type, year, month)
        
        # Scale features
        if energy_type in scalers:
            features_scaled = scalers[energy_type].transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        if energy_type in models:
            prediction = models[energy_type].predict(features_scaled)[0]
            return prediction, None
        else:
            return None, f"Model not found for {energy_type}"
            
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Energy Forecasting System</h1>', unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading models and data..."):
        historical_data = load_historical_data()
        models, scalers = load_models()
    
    if historical_data is None:
        st.error("Failed to load historical data. Please check the file path.")
        return
    
    # Sidebar for inputs
    st.sidebar.title("Prediction Parameters")
    
    # Energy type selection
    energy_types = {
        'Coal (Batu Bara)': 'coal',
        'Natural Gas (Gas Alam)': 'natural_gas',
        'Petroleum (Minyak Bumi)': 'petroleum',
        'Biodiesel': 'biodiesel',
        'Fuel Ethanol': 'fuel_ethanol'
    }
    
    selected_energy = st.sidebar.selectbox(
        "Select Energy Type:",
        list(energy_types.keys())
    )
    
    # Year and month selection
    current_year = datetime.now().year
    year = st.sidebar.number_input(
        "Target Year:",
        min_value=2024,
        max_value=2050,
        value=current_year,
        step=1
    )
    
    month = st.sidebar.selectbox(
        "Target Month:",
        list(range(1, 13)),
        format_func=lambda x: datetime(2023, x, 1).strftime('%B')
    )
    
    # Prediction button
    if st.sidebar.button("🔮 Make Prediction", type="primary"):
        energy_key = energy_types[selected_energy]
        
        with st.spinner("Making prediction..."):
            prediction, error = make_prediction(
                energy_key, year, month, models, scalers, historical_data
            )
        
        if error:
            st.error(f"Prediction failed: {error}")
        else:
            # Display results
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(
                    f'<div class="prediction-result">'
                    f'Predicted {selected_energy} Production<br>'
                    f'<span style="font-size: 2rem;">{prediction:,.2f}</span><br>'
                    f'for {datetime(year, month, 1).strftime("%B %Y")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    # Information section
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Lagging Features:**
        - **Biodiesel**: 1 year lag
        - **Other energies**: 12 months lag
        """)
    
    with col2:
        st.markdown("""
        **Available Models:**
        - Coal (SVR)
        - Natural Gas (SVR)  
        - Petroleum (SVR)
        - Biodiesel (Linear Regression)
        - Fuel Ethanol (SVR)
        """)
    
    # Historical data preview
    if st.checkbox("Show Historical Data Preview"):
        st.markdown("### 📈 Historical Data")
        
        # Show a sample of the data
        if historical_data is not None:
            # Display basic info about the dataset
            st.write(f"Dataset shape: {historical_data.shape}")
            st.dataframe(historical_data.head(10))
        else:
            st.warning("No historical data available")

if __name__ == "__main__":
    main()
