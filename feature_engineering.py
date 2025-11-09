import pandas as pd
import numpy as np

try:
    df = pd.read_csv('karachi_aqi_features.csv', parse_dates=True, index_col='time')
except FileNotFoundError:
    print("Error: 'karachi_aqi_features.csv' not found.")
    exit()

print("Starting feature engineering...")


df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 23.0)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 23.0)

df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 6.0)
df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 6.0)

df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12.0)


df['aqi_lag_1hr'] = df['AQI'].shift(1)
df['aqi_lag_3hr'] = df['AQI'].shift(3)
df['aqi_lag_24hr'] = df['AQI'].shift(24) 
df['aqi_lag_72hr'] = df['AQI'].shift(72) 

df['temp_lag_1hr'] = df['temperature_2m'].shift(1)
df['humidity_lag_1hr'] = df['relative_humidity_2m'].shift(1)
df['wind_speed_lag_1hr'] = df['wind_speed_10m'].shift(1)

print("Creating rolling window features...")

df['aqi_rolling_3hr'] = df['AQI'].shift(1).rolling(window=3).mean()

df['aqi_change_1hr'] = df['AQI'].shift(1) - df['AQI'].shift(2)



FORECAST_HORIZON = 72 # 3 days 
df['target_AQI'] = df['AQI'].shift(-FORECAST_HORIZON)


#dropping Nan rows
original_rows = len(df)
df_model_ready = df.dropna()
final_rows = len(df_model_ready)

print(f"\nOriginal rows: {original_rows}")
print(f"Rows dropped due to NaN (from lagging/shifting): {original_rows - final_rows}")
print(f"Final rows for modeling: {final_rows}")


columns_to_keep = [
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
    
    'temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m', 'wind_direction_10m',
    
    'aqi_lag_1hr', 'aqi_lag_3hr', 'aqi_lag_24hr', 'aqi_lag_72hr',
    
    'temp_lag_1hr', 'humidity_lag_1hr', 'wind_speed_lag_1hr',
    
    'aqi_rolling_3hr', 'aqi_change_1hr',
    'target_AQI'
]

df_model_ready = df_model_ready[columns_to_keep]

try:
    df_model_ready.to_csv('karachi_model_ready_data.csv')
    print(f"\nSuccessfully saved final model-ready data to 'karachi_model_ready_data.csv'")
    print("\n--- Final Data Head ---")
    print(df_model_ready.head())
except Exception as e:
    print(f"\nCould not save data to CSV: {e}")