import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



PM25_BREAKPOINTS = [
    ((0.0, 12.0), (0, 50)),
    ((12.1, 35.4), (51, 100)),
    ((35.5, 55.4), (101, 150)),
    ((55.5, 150.4), (151, 200)),
    ((150.5, 250.4), (201, 300)),
    ((250.5, 350.4), (301, 400)),
    ((350.5, 500.4), (401, 500)),
]

PM10_BREAKPOINTS = [
    ((0, 54), (0, 50)),
    ((55, 154), (51, 100)),
    ((155, 254), (101, 150)),
    ((255, 354), (151, 200)),
    ((355, 424), (201, 300)),
    ((425, 504), (301, 400)),
    ((505, 604), (401, 500)),
]

O3_BREAKPOINTS = [
    ((0, 106), (0, 50)),      # 0.000-0.054 ppm
    ((107, 137), (51, 100)),   # 0.055-0.070 ppm
    ((138, 167), (101, 150)),  # 0.071-0.085 ppm
    ((168, 206), (151, 200)),  # 0.086-0.105 ppm
    ((207, 392), (201, 300))   # 0.106-0.200 ppm
]

CO_BREAKPOINTS = [
    ((0, 5040), (0, 50)),       # 0.0-4.4 ppm
    ((5041, 10760), (51, 100)),   # 4.5-9.4 ppm
    ((10761, 14200), (101, 150)), # 9.5-12.4 ppm
    ((14201, 17600), (151, 200)), # 12.5-15.4 ppm
    ((17601, 34800), (201, 300)), # 15.5-30.4 ppm
    ((34801, 46300), (301, 400)), # 30.5-40.4 ppm
    ((46301, 57800), (401, 500)), # 40.5-50.4 ppm
]

SO2_BREAKPOINTS = [
    ((0, 92), (0, 50)),       # 0-35 ppb
    ((93, 197), (51, 100)),   # 36-75 ppb
    ((198, 482), (101, 150)), # 76-185 ppb
    ((483, 795), (151, 200)), # 186-304 ppb
]


NO2_BREAKPOINTS = [
    ((0, 100), (0, 50)),      # 0-53 ppb
    ((101, 188), (51, 100)),   # 54-100 ppb
    ((189, 677), (101, 150)),  # 101-360 ppb
    ((678, 1220), (151, 200)), # 361-649 ppb
    ((1221, 2330), (201, 300)),# 650-1249 ppb
    ((2331, 3080), (301, 400)),# 1250-1649 ppb
    ((3081, 3835), (401, 500)),# 1650-2049 ppb
]

def calculate_sub_index(concentration, breakpoints):
    """Calculates the AQI sub-index for a given concentration and breakpoint table."""
    if pd.isna(concentration):
        return np.nan
        
    for (conc_low, conc_high), (aqi_low, aqi_high) in breakpoints:
        if conc_low <= concentration <= conc_high:

            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (concentration - conc_low) + aqi_low
            return round(aqi)
            
    if concentration > breakpoints[-1][0][1]:
        return 500
    return np.nan 

try:
    df = pd.read_csv('karachi_aqi_weather_data.csv', parse_dates=True, index_col='time')
except FileNotFoundError:
    print("Error: 'karachi_aqi_weather_data.csv' not found.")
    exit()

print("Calculating AQI sub-indices...")
df['pm2_5_subindex'] = df['pm2_5'].apply(lambda x: calculate_sub_index(x, PM25_BREAKPOINTS))
df['pm10_subindex'] = df['pm10'].apply(lambda x: calculate_sub_index(x, PM10_BREAKPOINTS))
df['ozone_subindex'] = df['ozone'].apply(lambda x: calculate_sub_index(x, O3_BREAKPOINTS))
df['carbon_monoxide_subindex'] = df['carbon_monoxide'].apply(lambda x: calculate_sub_index(x, CO_BREAKPOINTS))
df['sulphur_dioxide_subindex'] = df['sulphur_dioxide'].apply(lambda x: calculate_sub_index(x, SO2_BREAKPOINTS))
df['nitrogen_dioxide_subindex'] = df['nitrogen_dioxide'].apply(lambda x: calculate_sub_index(x, NO2_BREAKPOINTS))

sub_indices_cols = [
    'pm2_5_subindex', 
    'pm10_subindex', 
    'ozone_subindex',
    'carbon_monoxide_subindex', 
    'sulphur_dioxide_subindex',
    'nitrogen_dioxide_subindex'
]

print("Calculating final AQI (max of all sub-indices)...")
df['AQI'] = df[sub_indices_cols].max(axis=1, skipna=True)

print("\n--- EDA: Missing Data Check ---")
print(df.isnull().sum())

# drop where no AQI 
df_cleaned = df.dropna(subset=['AQI'])

print("\n--- EDA: Plotting AQI Trend ---")
df_daily_avg = df_cleaned['AQI'].resample('D').mean()

plt.figure(figsize=(15, 6))
df_daily_avg.plot(title='Daily Average AQI in Karachi (2023-Present)')
plt.ylabel('Daily Average AQI')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()
plt.show()

print("Plot window opened. Please close it to continue.")

try:
    df_cleaned.to_csv('karachi_aqi_features.csv')
    print(f"\nSuccessfully saved enriched data to 'karachi_aqi_features.csv'")
except Exception as e:
    print(f"\nCould not save data to CSV: {e}")