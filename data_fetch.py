import requests
import pandas as pd
from io import StringIO


# Karachi Coordinates from Wikipedia: 24°51′36″N 67°0′36″E
latitude = 24.86
longitude = 67.01

start_date = "2023-01-01"
end_date = pd.to_datetime("today").strftime('%Y-%m-%d')

print(f"Fetching data (Lat: {latitude}, Lon: {longitude})")
print(f"Date Range: {start_date} to {end_date}\n")

air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    "timezone": "auto"
}

aq_response = requests.get(air_quality_url, params=aq_params)

if aq_response.status_code == 200:
    print("Air quality data fetched successfully.")
    aq_data = aq_response.json()
    
    aq_df = pd.DataFrame(aq_data['hourly'])
    aq_df['time'] = pd.to_datetime(aq_df['time'])
    aq_df.set_index('time', inplace=True)
else:
    print(f"Failed to fetch air quality data: {aq_response.status_code}")
    exit()

weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
    "timezone": "auto"
}

weather_response = requests.get(weather_url, params=weather_params)

if weather_response.status_code == 200:
    print("Weather data fetched successfully.")
    weather_data = weather_response.json()
    
    weather_df = pd.DataFrame(weather_data['hourly'])
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df.set_index('time', inplace=True)
else:
    print(f"Failed to fetch weather data: {weather_response.status_code}")
    exit()

print("\nCombining datasets...")
combined_df = aq_df.join(weather_df, how='inner')

print("\n--- Combined Data Head ---")
print(combined_df.head())

print("\n--- Data Info ---")
combined_df.info()

try:
    combined_df.to_csv("karachi_aqi_weather_data.csv")
    print(f"\nSuccessfully saved data to 'karachi_aqi_weather_data.csv'")
except Exception as e:
    print(f"\nCould not save data to CSV: {e}")