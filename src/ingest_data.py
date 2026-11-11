import sqlite3
import pandas as pd
import numpy as np
import datetime
import os

def generate_and_ingest_data():
    print("⏳ Simulating 1 year of hourly building environmental logs...")
    
    # Establish time horizon parameters (8,760 hours in a year)
    start_date = datetime.datetime(2026, 1, 1, 0, 0)
    timestamps = [start_date + datetime.timedelta(hours=i) for i in range(8760)]
    
    # Synthesize mathematical environmental cycles using sinusoidal distributions
    hours = np.array([t.hour for t in timestamps])
    days = np.array([t.timetuple().tm_yday for t in timestamps])
    
    # Ambient Temperature: Seasonal curve + Daily swing + Random noise
    seasonal_temp = 20 + 12 * np.sin(2 * np.pi * (days - 120) / 365)
    daily_swing = 5 * np.sin(2 * np.pi * (hours - 6) / 24)
    noise = np.random.normal(0, 1.5, len(timestamps))
    ambient_temp = seasonal_temp + daily_swing + noise
    
    # Relative Humidity: Inversely correlated with temperature
    humidity = np.clip(80 - (ambient_temp - 15) * 1.2 + np.random.normal(0, 3, len(timestamps)), 15, 98)
    
    # Estimated Occupancy Load: Office building profile (Weekdays vs Weekends)
    occupancy = []
    for t in timestamps:
        if t.weekday() < 5:  # Weekday
            if 8 <= t.hour <= 18:
                occupancy.append(int(np.random.normal(850, 50)))
            elif 6 <= t.hour < 8 or 18 < t.hour <= 21:
                occupancy.append(int(np.random.normal(20, 30)))
            else:
                occupancy.append(int(np.random.normal(15, 5)))
        else:  # Weekend
            occupancy.append(int(np.random.normal(20, 5)) if 9 <= t.hour <= 17 else int(np.random.normal(5, 2)))
    occupancy = np.clip(occupancy, 0, 1200).astype(int)
    
    # Calculate Energy Consumption Load matrices (kW)
    cooling_load = np.maximum(0, ambient_temp - 22) * 12.5
    heating_load = np.maximum(0, 15 - ambient_temp) * 8.0
    internal_gain = occupancy * 0.15
    base_load = 150 + cooling_load + heating_load + internal_gain + np.random.normal(0, 15, len(timestamps))
    
    # Optimization load achieves a 15-20% load flattening shave during peak hours
    optimized_load = base_load.copy()
    
    # Quick standard boolean filter mask array mapping loops
    is_weekday = np.array([t.weekday() < 5 for t in timestamps])
    peak_hours = (hours >= 9) & (hours <= 17) & is_weekday
    optimized_load[peak_hours] = base_load[peak_hours] * np.random.uniform(0.80, 0.85, sum(peak_hours))
    
    # Construct structured DataFrame frame
    df = pd.DataFrame({
        'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
        'ambient_temperature': np.round(ambient_temp, 2),
        'relative_humidity': np.round(humidity, 2),
        'occupancy_count': occupancy,
        'baseline_load_kw': np.round(base_load, 2),
        'optimized_load_kw': np.round(optimized_load, 2)
    })
    
    # Commit dataset to local SQL Warehouse
    db_path = os.path.join('data', 'hvac_optimization.db')
    conn = sqlite3.connect(db_path)
    
    print("📥 Streaming data packets into 'building_telemetry' SQL table matrix...")
    df.to_sql('building_telemetry', conn, if_exists='replace', index=False)
    
    conn.commit()
    conn.close()
    print(f"✅ Ingestion successful! 8,760 hourly logs populated safely in the database.")

if __name__ == '__main__':
    generate_and_ingest_data()
