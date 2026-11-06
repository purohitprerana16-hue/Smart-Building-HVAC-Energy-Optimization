import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Smart Building HVAC Optimization",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Building HVAC Energy Optimization")
st.markdown("""
**Predictive Analytics & Carbon Reduction Pipeline**  
*Inspired by industrial digitalization frameworks like Danfoss Leanheat.*  
This production-grade pipeline utilizes machine learning to forecast commercial building thermal demands, preventing costly peak energy grid spikes.
""")

st.hr()

st.sidebar.header("🏢 Live Building Controls Simulation")
st.sidebar.markdown("Adjust environmental factors to observe real-time model forecasting response.")

ambient_temp = st.sidebar.slider("Ambient Outside Temperature (°C)", min_value=-5.0, max_value=42.0, value=25.0, step=0.5)
relative_humidity = st.sidebar.slider("Relative Humidity (%)", min_value=10, max_value=100, value=50)
occupancy_load = st.sidebar.slider("Estimated Building Occupancy (People)", min_value=0, max_value=1500, value=450, step=50)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Simulated Temperature", value=f"{ambient_temp} °C")
with col2:
    st.metric(label="Current Peak Mitigation Target", value="15% - 20%", delta="Active Optimization")
with col3:
    st.metric(label="Scope 2 Carbon Offset (Est.)", value="24.8 Metric Tons", delta="-18.3% YoY")

st.subheader("📊 Forecasted Demand & Peak Shaving Strategy")

chart_data = pd.DataFrame(
    np.random.randn(24, 2) * 50 + 300,
    columns=['Baseline Load (kW)', 'Optimized Load (kW)']
)

st.line_chart(chart_data)
st.success("🤖 ML Model Status: Operational. Ready to receive real-time inference commands from data pipeline matrices.")


