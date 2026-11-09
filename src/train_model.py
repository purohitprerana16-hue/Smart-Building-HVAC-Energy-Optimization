import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_predictive_engine():
    db_path = os.path.join('data', 'hvac_optimization.db')
    model_dir = os.path.join('src', 'models')
    model_path = os.path.join(model_dir, 'hvac_model.pkl')
    
    print("🔌 Extracting historical building logs from SQL relational warehouse...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM building_telemetry", conn)
    conn.close()
    
    print("🧠 Engineering temporal feature arrays for model training constraints...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    
    X = df[['ambient_temperature', 'relative_humidity', 'occupancy_count', 'hour', 'month']]
    y = df['baseline_load_kw']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🤖 Training predictive machine learning model (Random Forest Regressor)...")
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"📊 Validation Engine Metrics: MAE = {mae:.2f} kW | R² Accuracy Score = {r2:.4f}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"✅ AI Predictive Engine successfully trained and exported live to: {model_path}")

if __name__ == '__main__':
    train_predictive_engine()
