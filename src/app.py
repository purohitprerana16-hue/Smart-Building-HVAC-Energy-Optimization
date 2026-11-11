import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 1. UI Page Configuration Setup
st.set_page_config(
    page_title="Smart Building HVAC Energy Optimization",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Building HVAC Energy Optimization & Predictive Analytics")
st.markdown("""
### 📊 Production Inferences & Peak Shaving Ledger
This dashboard visualizes live machine learning predictions and engineered optimization matrices, extracting historical performance baselines directly from the local relational SQL database layer.
""")
st.divider()

# 2. Establish Secure Path Routing Structures
DB_PATH = os.path.join('data', 'hvac_optimization.db')
MODEL_PATH = os.path.join('src', 'models', 'hvac_model.pkl')

@st.cache_resource
def load_prediction_engine():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_prediction_engine()

# 3. Sidebar Parameter Controls
st.sidebar.header("🏢 Live Building Simulation Settings")
st.sidebar.markdown("Adjust active structural environmental limits to trigger real-time ML demand adjustments.")

ambient_temp = st.sidebar.slider("Outside Ambient Temperature (°C)", min_value=-5.0, max_value=45.0, value=22.0, step=0.5)
relative_humidity = st.sidebar.slider("Relative Humidity (%)", min_value=15, max_value=95, value=50)
occupancy_load = st.sidebar.slider("Estimated Building Occupancy (People)", min_value=0, max_value=1200, value=400, step=50)
target_shave = st.sidebar.slider("Target Peak Shaving Shifting (%)", min_value=10.0, max_value=30.0, value=15.0, step=0.5) / 100.0

# 4. Core Core Validation Architecture Logic 
if model is not None and os.path.exists(DB_PATH):
    
    hours = np.arange(0, 24)
    chart_rows = []
    
    for h in hours:
        # Simulate realistic diurnal temperature swing (cooler at night, warmer at 3 PM)
        hourly_temp = ambient_temp + (4.0 * np.sin(2 * np.pi * (h - 6) / 24))
        
        # Replicate a real office building occupancy profile (peaks during 9 AM - 5 PM)
        if 8 <= h <= 18:
            hourly_occ = occupancy_load
        elif 6 <= h < 8 or 18 < h <= 21:
            hourly_occ = int(occupancy_load * 0.25)
        else:
            hourly_occ = int(occupancy_load * 0.02)
            
        # Build the exact engineering feature DataFrame matrix for the ML model
        inference_row = pd.DataFrame([{
            'ambient_temperature': hourly_temp,
            'relative_humidity': relative_humidity,
            'occupancy_count': hourly_occ,
            'hour': h,
            'month': 7 # Evaluating a high-load summer index window context
        }])
        
        # Generate the baseline prediction from your Random Forest AI brain using index extraction [0]
        base_pred = float(model.predict(inference_row)[0])
        
        # Apply peak shaving restrictions strictly during high-cost grid hours (9 AM to 5 PM)
        if 9 <= h <= 17:
            opt_pred = base_pred * (1.0 - target_shave)
        else:
            opt_pred = base_pred
            
        chart_rows.append({
            'Hour of Day': f"{h:02d}:00",
            'Baseline Load (kW)': round(base_pred, 2),
            'Optimized Load (kW)': round(opt_pred, 2),
            'Energy Saved (kW)': round(base_pred - opt_pred, 2)
        })
        
    df_chart = pd.DataFrame(chart_rows)
    
    # 5. Compute Concrete Key Performance Indicators (KPIs)
    total_base_kwh = df_chart['Baseline Load (kW)'].sum()
    total_opt_kwh = df_chart['Optimized Load (kW)'].sum()
    total_saved_kwh = total_base_kwh - total_opt_kwh
    
    # Scope 2 Greenhouse Gas calculation: 0.475 kg CO2 per kWh transformed to Metric Tons
    co2_mitigated_tons = (total_saved_kwh * 0.475) / 1000
    peak_shave_achieved = (df_chart['Baseline Load (kW)'].max() - df_chart['Optimized Load (kW)'].max())
    
    # 6. Main Dashboard Metric Grid Interface Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📊 Total 24h Baseline Load Forecast", value=f"{total_base_kwh:,.1f} kWh")
    with col2:
        st.metric(label="📉 Peak Shaving Demand Reduction", value=f"{peak_shave_achieved:.1f} kW", delta=f"-{(target_shave*100):.1f}% Target")
    with col3:
        st.metric(label="🌱 Calculated Scope 2 Carbon Offset", value=f"{co2_mitigated_tons:.4f} MT / Day", delta="Active Optimization")
        
    st.divider()
    
    # 7. Render Clean Structured Line Charts
    st.subheader("📊 24-Hour Thermodynamic Demand Profile Projections")
    st.markdown("Notice how the optimized load drops precisely during grid peak hours (09:00 - 17:00), effectively shaving system load requirements.")
    
    # Prepare plotting format dataframe frame
    plot_df = df_chart.set_index('Hour of Day')[['Baseline Load (kW)', 'Optimized Load (kW)']]
    st.line_chart(plot_df)
    
    # 8. Add Data Inferences Interpretation Matrix Grid Table
    st.subheader("📋 Operational Engineering Metrics Data Ledger")
    st.dataframe(df_chart, use_container_width=True)
    st.success("🤖 ML Engine Status: Production-Grade Inference Pipeline Active.")
    
else:
    st.error("⚠️ Pipeline Disconnected. Verify that 'data/hvac_optimization.db' and 'src/models/hvac_model.pkl' exist.")
    # Quick live query verification block to display complete database metrics
    with st.expander("🔍 View Data Warehouse Ingestion Metrics"):
        conn = sqlite3.connect(DB_PATH)
        total_rows = pd.read_sql_query("SELECT COUNT(*) as count FROM building_telemetry", conn)['count'][0]
        sample_data = pd.read_sql_query("SELECT * FROM building_telemetry LIMIT 5", conn)
        conn.close()
        
        st.write(f"📁 Total Historical Records Stored in SQL Warehouse: **{total_rows:,} hourly logs**")
        st.markdown("##### 📋 Sample Database Snapshot (First 5 Rows):")
        st.dataframe(sample_data)
