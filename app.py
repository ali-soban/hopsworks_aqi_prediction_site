import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
from io import StringIO
import warnings
import datetime # Import datetime

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Page Configuration ---
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="💨",
    layout="wide"
)

# --- AQI Calculation Constants and Functions ---
# (Copied from our data processing script)
PM25_BREAKPOINTS = [
    ((0.0, 12.0), (0, 50)), ((12.1, 35.4), (51, 100)), ((35.5, 55.4), (101, 150)),
    ((55.5, 150.4), (151, 200)), ((150.5, 250.4), (201, 300)), ((250.5, 350.4), (301, 400)),
    ((350.5, 500.4), (401, 500)),
]
PM10_BREAKPOINTS = [
    ((0, 54), (0, 50)), ((55, 154), (51, 100)), ((155, 254), (101, 150)),
    ((255, 354), (151, 200)), ((355, 424), (201, 300)), ((425, 504), (301, 400)),
    ((505, 604), (401, 500)),
]
O3_BREAKPOINTS = [
    ((0, 106), (0, 50)), ((107, 137), (51, 100)), ((138, 167), (101, 150)),
    ((168, 206), (151, 200)), ((207, 392), (201, 300))
]
CO_BREAKPOINTS = [
    ((0, 5040), (0, 50)), ((5041, 10760), (51, 100)), ((10761, 14200), (101, 150)),
    ((14201, 17600), (151, 200)), ((17601, 34800), (201, 300)), ((34801, 46300), (301, 400)),
    ((46301, 57800), (401, 500)),
]
SO2_BREAKPOINTS = [
    ((0, 92), (0, 50)), ((93, 197), (51, 100)), ((198, 482), (101, 150)),
    ((483, 795), (151, 200)),
]
NO2_BREAKPOINTS = [
    ((0, 100), (0, 50)), ((101, 188), (51, 100)), ((189, 677), (101, 150)),
    ((678, 1220), (151, 200)), ((1221, 2330), (201, 300)), ((2331, 3080), (301, 400)),
    ((3081, 3835), (401, 500)),
]

def calculate_sub_index(concentration, breakpoints):
    """Calculates the AQI sub-index for a given concentration and breakpoint table."""
    # Ensure concentration is numeric, return NaN otherwise
    if not isinstance(concentration, (int, float, np.number)) or pd.isna(concentration):
        return np.nan

    for i, ((conc_low, conc_high), (aqi_low, aqi_high)) in enumerate(breakpoints):
        # Handle the first breakpoint range including the lower bound
        if i == 0 and conc_low <= concentration <= conc_high:
             # Avoid division by zero if conc_low == conc_high (shouldn't happen in standard tables)
            if conc_high == conc_low:
                return float(aqi_low)
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (concentration - conc_low) + aqi_low
            return round(aqi)
        # Handle subsequent ranges (exclusive of lower bound, inclusive of upper)
        elif i > 0 and conc_low < concentration <= conc_high:
             # Avoid division by zero
            if conc_high == conc_low:
                 return float(aqi_low) # Or handle as error/specific value
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (concentration - conc_low) + aqi_low
            return round(aqi)

    # Handle cases where concentration is off the charts (use highest AQI)
    # Check bounds before accessing index -1
    if breakpoints and concentration > breakpoints[-1][0][1]:
        # Check if the highest breakpoint range exists and has valid AQI values
        if len(breakpoints[-1]) > 1 and len(breakpoints[-1][1]) > 1:
            return breakpoints[-1][1][1] # Return the max AQI value for that pollutant range
        else:
             return 500 # Default max AQI if table is malformed

    # Handle cases where concentration is below the lowest breakpoint (e.g., negative)
    # Check bounds before accessing index 0
    if breakpoints and len(breakpoints[0]) > 0 and len(breakpoints[0][0]) > 0:
        if concentration < breakpoints[0][0][0]:
             # Check if the lowest breakpoint range exists and has valid AQI values
            if len(breakpoints[0]) > 1 and len(breakpoints[0][1]) > 0:
                return breakpoints[0][1][0] # Return the min AQI value (usually 0)
            else:
                 return 0 # Default min AQI

    return np.nan # If no condition met (should be rare with valid inputs/tables)


def get_aqi_category(aqi):
    """Returns the AQI category and color."""
    if pd.isna(aqi):
        return "Unknown", "#808080" # Gray for NaN
    try:
        aqi = int(aqi) # Convert to integer after checking for NaN
    except (ValueError, TypeError):
         return "Invalid", "#808080" # Handle non-numeric after NaN check

    if aqi <= 50:
        return "Good", "#00E400"  # Green
    elif aqi <= 100:
        return "Moderate", "#FFFF00"  # Yellow
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"  # Orange
    elif aqi <= 200:
        return "Unhealthy", "#FF0000"  # Red
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97"  # Purple
    elif aqi > 300:
        return "Hazardous", "#7E0023"  # Maroon
    else: # Should ideally not be reached
        return "Unknown", "#808080"  # Gray

# --- Feature Engineering Function ---
def create_features(df_raw):
    """Runs the full AQI calculation and feature engineering pipeline."""
    df = df_raw.copy() # Work on a copy

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("DataFrame index is not a DatetimeIndex in create_features.")
        return None, None

    # Check timezone awareness BEFORE calculations involving index time
    # st.write(f"Index timezone in create_features: {df.index.tz}") # Debugging line

    # 1. Calculate AQI
    # Convert pollutant columns to numeric first, coercing errors
    pollutant_cols = ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If a pollutant column is missing entirely, add it with NaNs
            df[col] = np.nan

    df['pm2_5_subindex'] = df['pm2_5'].apply(lambda x: calculate_sub_index(x, PM25_BREAKPOINTS))
    df['pm10_subindex'] = df['pm10'].apply(lambda x: calculate_sub_index(x, PM10_BREAKPOINTS))
    df['ozone_subindex'] = df['ozone'].apply(lambda x: calculate_sub_index(x, O3_BREAKPOINTS))
    df['carbon_monoxide_subindex'] = df['carbon_monoxide'].apply(lambda x: calculate_sub_index(x, CO_BREAKPOINTS))
    df['sulphur_dioxide_subindex'] = df['sulphur_dioxide'].apply(lambda x: calculate_sub_index(x, SO2_BREAKPOINTS))
    df['nitrogen_dioxide_subindex'] = df['nitrogen_dioxide'].apply(lambda x: calculate_sub_index(x, NO2_BREAKPOINTS))

    sub_indices_cols = ['pm2_5_subindex', 'pm10_subindex', 'ozone_subindex',
                        'carbon_monoxide_subindex', 'sulphur_dioxide_subindex', 'nitrogen_dioxide_subindex']
    # Ensure sub-indices are numeric before max()
    for col in sub_indices_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['aqi'] = df[sub_indices_cols].max(axis=1, skipna=True)

    # Make a copy for feature engineering AFTER calculating AQI on the full set
    df_features = df.copy()

    # 2. Create Time-Based (Cyclical) Features (using index properties)
    # Make sure index exists and is datetime before accessing properties
    if isinstance(df_features.index, pd.DatetimeIndex):
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features.index.hour / 23.0)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features.index.hour / 23.0)
        df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features.index.dayofweek / 6.0)
        df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features.index.dayofweek / 6.0)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    else:
        st.error("Index is not DatetimeIndex before creating cyclical features.")
        # Handle error or return None, depending on desired behavior
        return None, None


    # 3. Create Lagged Features
    # Ensure relevant columns exist before lagging
    weather_cols_to_lag = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
    for col in weather_cols_to_lag:
        if col not in df_features.columns:
             df_features[col] = np.nan # Add missing weather cols as NaN if needed

    df_features['aqi_lag_1hr'] = df_features['aqi'].shift(1)
    df_features['aqi_lag_3hr'] = df_features['aqi'].shift(3)
    df_features['aqi_lag_24hr'] = df_features['aqi'].shift(24)
    df_features['aqi_lag_72hr'] = df_features['aqi'].shift(72)
    df_features['temp_lag_1hr'] = df_features['temperature_2m'].shift(1)
    df_features['humidity_lag_1hr'] = df_features['relative_humidity_2m'].shift(1)
    df_features['wind_speed_lag_1hr'] = df_features['wind_speed_10m'].shift(1)

    # 4. Create Rolling Window & Rate of Change Features
    df_features['aqi_rolling_3hr'] = df_features['aqi'].shift(1).rolling(window=3).mean()
    df_features['aqi_change_1hr'] = df_features['aqi'].shift(1) - df_features['aqi'].shift(2)

    # Return the dataframe with calculated AQI (for current value) and the feature-engineered dataframe
    # Ensure both returned dataframes retain the original index timezone if it was present
    return df, df_features


# --- Data Loading and Caching ---
@st.cache_resource
def load_model(model_path='aqi_model_extratrees.joblib'):
    """Loads the trained model, caching it to avoid reloading."""
    try:
        model = joblib.load(model_path)
        st.session_state['model_loaded'] = True # Flag to indicate model is loaded
        # Store expected feature names from the loaded model if available
        if hasattr(model, 'feature_names_in_'):
             st.session_state['model_features'] = model.feature_names_in_
        else:
             # Fallback if the attribute doesn't exist (e.g., older scikit-learn)
             # Define the expected columns manually - MUST MATCH TRAINING
             st.session_state['model_features'] = [
                'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
                'temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m', 'wind_direction_10m',
                'aqi_lag_1hr', 'aqi_lag_3hr', 'aqi_lag_24hr', 'aqi_lag_72hr',
                'temp_lag_1hr', 'humidity_lag_1hr', 'wind_speed_lag_1hr',
                'aqi_rolling_3hr', 'aqi_change_1hr'
             ]
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Please place it in the same directory as the app.")
        st.session_state['model_loaded'] = False
        st.session_state['model_features'] = None
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        st.session_state['model_loaded'] = False
        st.session_state['model_features'] = None
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour (3600 seconds)
def get_forecast_data():
    """Fetches and processes the latest data for forecasting."""
    st.write("Fetching latest data from APIs (using UTC)...") # Progress indicator
    # 1. Define Coordinates and Time Range
    latitude = 24.86
    longitude = 67.01
    fetch_days_past = 10 # Need enough history for lags (72 hours = 3 days, plus buffer)
    fetch_days_future = 3 # Fetch weather forecast for the prediction period

    # Use UTC for consistency
    now_utc = pd.to_datetime("now", utc=True)

    start_date_aq = (now_utc - pd.Timedelta(days=fetch_days_past)).strftime('%Y-%m-%d')
    end_date_aq = now_utc.strftime('%Y-%m-%d') # Fetch AQ up to 'today' (UTC)

    start_date_weather = start_date_aq
    end_date_weather = (now_utc + pd.Timedelta(days=fetch_days_future)).strftime('%Y-%m-%d') # Fetch weather forecast including future (UTC)

    # 2. Fetch Air Quality Data (Past only, in UTC)
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start_date_aq, "end_date": end_date_aq,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "UTC" # Explicitly request UTC
    }
    try:
        aq_response = requests.get(air_quality_url, params=aq_params)
        aq_response.raise_for_status()
        aq_data = aq_response.json()
        aq_df = pd.DataFrame(aq_data['hourly'])
        aq_df['time'] = pd.to_datetime(aq_df['time'], utc=True) # Parse as UTC
        aq_df.set_index('time', inplace=True)

    except Exception as e:
        st.error(f"Failed to fetch air quality data: {e}")
        return None, None, None, None # Added None for latest_aqi_time_utc

    # 3. Fetch Weather Data (Archive for past, Forecast for future, in UTC)
    weather_url_archive = "https://archive-api.open-meteo.com/v1/archive"
    weather_url_forecast = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude, "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
        "timezone": "UTC" # Explicitly request UTC
    }

    all_weather_dfs = []
    try:
        # Fetch past data up to yesterday (UTC)
        end_date_past_weather = (now_utc - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if start_date_weather <= end_date_past_weather:
            past_params = {**weather_params, "start_date": start_date_weather, "end_date": end_date_past_weather}
            past_weather_response = requests.get(weather_url_archive, params=past_params)
            past_weather_response.raise_for_status()
            past_weather_data = past_weather_response.json()
            if 'hourly' in past_weather_data and past_weather_data['hourly']:
                 past_weather_df = pd.DataFrame(past_weather_data['hourly'])
                 # Ensure 'time' column exists before converting
                 if 'time' in past_weather_df.columns:
                     past_weather_df['time'] = pd.to_datetime(past_weather_df['time'], utc=True) # Parse as UTC
                     all_weather_dfs.append(past_weather_df)
                 else:
                      st.warning("Missing 'time' column in past weather data.")


        # Fetch future data (including today, UTC)
        # --- FIX 1 START ---
        # Convert end_date_weather string to UTC timestamp for comparison
        end_date_weather_ts = pd.to_datetime(end_date_weather, utc=True)
        # --- FIX 1 END ---
        forecast_days_needed = (end_date_weather_ts - now_utc.normalize()).days + 1
        future_params = {**weather_params, "forecast_days": max(1, forecast_days_needed)} # Ensure at least 1 day

        future_weather_response = requests.get(weather_url_forecast, params=future_params)
        future_weather_response.raise_for_status()
        future_weather_data = future_weather_response.json()
        if 'hourly' in future_weather_data and future_weather_data['hourly']:
            future_weather_df = pd.DataFrame(future_weather_data['hourly'])
            # Ensure 'time' column exists before converting
            if 'time' in future_weather_df.columns:
                future_weather_df['time'] = pd.to_datetime(future_weather_df['time'], utc=True) # Parse as UTC
                all_weather_dfs.append(future_weather_df)
            else:
                 st.warning("Missing 'time' column in future weather data.")


        # Combine weather data
        if not all_weather_dfs:
             raise ValueError("No weather data could be fetched or processed.")

        # Set index after concatenation
        weather_df_combined = pd.concat(all_weather_dfs, ignore_index=True)
        if 'time' not in weather_df_combined.columns:
             raise ValueError("'time' column missing after concatenating weather data.")

        weather_df_combined.set_index('time', inplace=True)
        weather_df = weather_df_combined[~weather_df_combined.index.duplicated(keep='first')] # Remove any overlap
        weather_df = weather_df.sort_index() # Ensure chronological order

    except Exception as e:
        st.error(f"Failed to fetch or process weather data: {e}")
        return None, None, None, None

    # 4. Combine AQ and Weather Data
    df_combined = aq_df.join(weather_df, how='outer')
    df_combined = df_combined.sort_index()
    # Ensure index is timezone-aware (UTC) after join
    # --- REFINED FIX 2 START ---
    if df_combined.index.tz is None:
         df_combined = df_combined.tz_localize('UTC')
    # Check if tz exists and is not UTC
    elif df_combined.index.tz is not None and df_combined.index.tz != datetime.timezone.utc:
         df_combined = df_combined.tz_convert('UTC')
    # --- REFINED FIX 2 END ---


    #st.write("Combined DataFrame Head:") # Debug
    #st.dataframe(df_combined.head()) # Debug
    #st.write(f"Combined DataFrame Index TZ: {df_combined.index.tz}") # Debug

    st.write("Data fetched. Processing features...") # Progress indicator

    # 5. Process Features
    df_with_aqi, df_features = create_features(df_combined)
    if df_with_aqi is None or df_features is None:
         st.error("Feature creation failed.")
         return None, None, None, None

    #st.write("Features DataFrame Head:") # Debug
    #st.dataframe(df_features.head()) # Debug
    #st.write(f"Features DataFrame Index TZ: {df_features.index.tz}") # Debug


    # 6. Get current AQI (using UTC now for comparison)
    # Ensure now_utc is timezone-aware
    now_utc = pd.Timestamp.now(tz='UTC')
    # Ensure the index is timezone-aware for comparison
    df_with_aqi_utc = df_with_aqi
    # --- REFINED FIX 3 START ---
    if df_with_aqi_utc.index.tz is None:
        st.warning("df_with_aqi index became naive, attempting to localize to UTC.")
        try:
             df_with_aqi_utc = df_with_aqi_utc.tz_localize('UTC')
        except TypeError: # Already localized
             # Check if it's already UTC, if not, convert
             if df_with_aqi_utc.index.tz is not None and df_with_aqi_utc.index.tz != datetime.timezone.utc:
                 df_with_aqi_utc = df_with_aqi_utc.tz_convert('UTC')
    # Check if tz exists and is not UTC
    elif df_with_aqi_utc.index.tz is not None and df_with_aqi_utc.index.tz != datetime.timezone.utc:
         df_with_aqi_utc = df_with_aqi_utc.tz_convert('UTC')
    # --- REFINED FIX 3 END ---


    # Find the last index time <= now_utc where AQI is valid
    valid_aqi_times = df_with_aqi_utc['aqi'].dropna().index
    if not valid_aqi_times.empty:
         # Filter times before or equal to now_utc
         relevant_times = valid_aqi_times[valid_aqi_times <= now_utc]
         if not relevant_times.empty:
             latest_available_time_utc = relevant_times.max()
         else:
              latest_available_time_utc = pd.NaT # No valid AQI time found before now
    else:
         latest_available_time_utc = pd.NaT


    if latest_available_time_utc is pd.NaT:
         current_aqi = np.nan
         st.warning("Could not determine the latest available AQI time.")
    else:
        try:
            current_aqi = df_with_aqi_utc.loc[latest_available_time_utc, 'aqi']
        except KeyError:
            current_aqi = np.nan
            st.warning(f"Timestamp {latest_available_time_utc} found but not in index for AQI lookup.")
        #st.write(f"Latest AQI time (UTC): {latest_available_time_utc}, Value: {current_aqi}") # Debug


    # 7. Prepare data for prediction (using UTC now)
    forecast_start_time_utc = now_utc.ceil('h') # Start forecast from the next whole hour in UTC

    # The input for the *first* prediction uses data from the hour *before* the forecast starts
    last_known_data_time_utc = forecast_start_time_utc - pd.Timedelta(hours=1)

    # Ensure df_features index is UTC for lookup
    # --- REFINED FIX 4 START ---
    if df_features.index.tz is None:
         st.warning("df_features index became naive, attempting to localize to UTC.")
         try:
            df_features = df_features.tz_localize('UTC')
         except TypeError: # Already localized
            # Check if it's already UTC, if not, convert
            if df_features.index.tz is not None and df_features.index.tz != datetime.timezone.utc:
                df_features = df_features.tz_convert('UTC')
    # Check if tz exists and is not UTC
    elif df_features.index.tz is not None and df_features.index.tz != datetime.timezone.utc:
         df_features = df_features.tz_convert('UTC')
    # --- REFINED FIX 4 END ---


    if last_known_data_time_utc not in df_features.index:
         st.warning(f"Required data point for prediction ({last_known_data_time_utc}) not found in processed features. Index range: {df_features.index.min()} to {df_features.index.max()}")
         # Try using the absolute latest available index if the exact previous hour is missing
         fallback_time_utc = df_features.index.max()
         if fallback_time_utc is not pd.NaT and fallback_time_utc < forecast_start_time_utc:
              st.warning(f"Falling back to using latest available data at {fallback_time_utc} for prediction.")
              last_known_data_time_utc = fallback_time_utc
         else:
              st.error("Cannot find a suitable recent data point to start the forecast.")
              return current_aqi, None, None, latest_available_time_utc

    # Get the feature columns expected by the model
    feature_cols = st.session_state.get('model_features')
    if feature_cols is None:
         st.error("Model feature names not found. Cannot prepare prediction input.")
         return current_aqi, None, None, latest_available_time_utc

    # Ensure all required feature columns exist in df_features, add if missing (with NaN)
    for col in feature_cols:
         if col not in df_features.columns:
             st.warning(f"Feature column '{col}' missing from processed data, adding as NaN.")
             df_features[col] = np.nan

    # Select the specific row needed to predict the first hour
    try:
        # Reindex ensures all expected columns are present, even if they were NaN before selection
        X_forecast_input_row = df_features.loc[[last_known_data_time_utc]].reindex(columns=list(feature_cols))
    except KeyError:
         st.error(f"Timestamp {last_known_data_time_utc} not found in index after checks.")
         return current_aqi, None, None, latest_available_time_utc
    except Exception as e:
         st.error(f"Error selecting prediction input row: {e}")
         return current_aqi, None, None, latest_available_time_utc


    # Check for NaNs in the input row *before* prediction
    if X_forecast_input_row.isnull().values.any():
        st.warning("Missing values detected in the features needed for the prediction. Cannot generate forecast.")
        st.dataframe(X_forecast_input_row.isnull().sum()[X_forecast_input_row.isnull().sum() > 0]) # Show missing columns
        return current_aqi, None, None, latest_available_time_utc

    st.write("Features processed.") # Progress indicator
    return current_aqi, X_forecast_input_row, df_features, latest_available_time_utc # Return the single row, full df, and time


# --- Main Application ---
st.title("💨 Pearls AQI Predictor: Karachi")

# Initialize session state for model loading and features
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'model_features' not in st.session_state:
    st.session_state['model_features'] = None

# Display loading spinner while model loads
with st.spinner('Loading prediction model...'):
    model = load_model()

if not st.session_state['model_loaded']:
    st.error("Model could not be loaded. Cannot proceed.")
    st.stop()
# No need for success message here, proceed if loaded

# Add a button to refresh data and run forecast
if st.button("🔄 Refresh Forecast"):
    # Clear cache before fetching new data
    st.cache_data.clear()
    st.cache_resource.clear()
    # Use st.rerun() which is standard now
    st.rerun()

# Display timestamp in local Karachi time
try:
    karachi_tz = 'Asia/Karachi'
    karachi_now = pd.to_datetime('now', utc=True).tz_convert(karachi_tz)
    display_time_str = karachi_now.strftime('%Y-%m-%d %H:%M %Z')
except Exception as e:
    st.warning(f"Could not convert time to Karachi timezone ({karachi_tz}): {e}")
    display_time_str = pd.to_datetime('now', utc=True).strftime('%Y-%m-%d %H:%M %Z')

st.markdown(f"Using a machine learning model to forecast the Air Quality Index (AQI) 72 hours in advance. _(Data fetched around: {display_time_str})_")


# Get Data and Features with progress indication
current_aqi = np.nan
X_forecast_input_row = None
df_features_full = None
latest_available_time_utc = pd.NaT

with st.spinner('Fetching latest data and generating features...'):
    current_aqi, X_forecast_input_row, df_features_full, latest_available_time_utc = get_forecast_data()

# --- Display Section ---
# Try to display current AQI regardless of forecast success
current_aqi_display_val = "N/A"
current_cat_display = "Unknown"
current_col_display = "#808080"
current_time_display = "Unknown Time"

if latest_available_time_utc is not pd.NaT and not pd.isna(current_aqi):
    current_aqi_display_val = int(current_aqi)
    current_cat_display, current_col_display = get_aqi_category(current_aqi)
    try:
        current_time_local = latest_available_time_utc.tz_convert(karachi_tz)
        current_time_display = current_time_local.strftime('%Y-%m-%d %H:%M %Z')
    except Exception as e:
        st.warning(f"Could not convert latest AQI time to local: {e}")
        current_time_display = latest_available_time_utc.strftime('%Y-%m-%d %H:%M %Z') + " (UTC)"


# Make Prediction and Display Forecast if possible
predicted_aqi_72h = "N/A"
forecast_cat_display = "Unknown"
forecast_col_display = "#808080"
forecast_time_display = "Unknown Time"
forecast_time_utc = pd.NaT # Initialize forecast_time_utc
can_forecast = X_forecast_input_row is not None and model is not None and st.session_state.get('model_features') is not None

if can_forecast:
    try:
        with st.spinner('Generating 72-hour forecast...'):
            # Ensure the input dataframe columns match the model's expected features
            # Use reindex to guarantee order and presence of columns, fill missing with NaN (shouldn't happen due to earlier checks)
            X_input_aligned = X_forecast_input_row.reindex(columns=st.session_state['model_features'])

            # Final check for NaNs after alignment
            if X_input_aligned.isnull().values.any():
                st.warning("Missing values detected after aligning input features. Cannot forecast.")
                st.dataframe(X_input_aligned.isnull().sum()[X_input_aligned.isnull().sum() > 0])
                can_forecast = False
            else:
                y_forecast_values = model.predict(X_input_aligned)
                predicted_aqi_72h = int(round(y_forecast_values[0]))
                forecast_cat_display, forecast_col_display = get_aqi_category(predicted_aqi_72h)

                # Calculate forecast time in UTC then convert to local for display
                forecast_time_utc = X_forecast_input_row.index[0] + pd.Timedelta(hours=72)
                try:
                    forecast_time_local = forecast_time_utc.tz_convert(karachi_tz)
                    forecast_time_display = forecast_time_local.strftime('%Y-%m-%d %H:%M %Z')
                except Exception as e:
                     st.warning(f"Could not convert forecast time to local: {e}")
                     forecast_time_display = forecast_time_utc.strftime('%Y-%m-%d %H:%M %Z') + " (UTC)"

    except Exception as e:
        st.error(f"An error occurred during model prediction: {e}")
        st.exception(e) # Show full traceback for debugging
        can_forecast = False # Mark forecast as failed

# --- Display UI Elements ---
st.subheader("AQI Forecast")

# --- Hazardous AQI Alert ---
try:
    # Use 0 for comparison if current AQI is NA, predict val must be numeric
    current_aqi_comp = current_aqi if not pd.isna(current_aqi) else 0
    predict_aqi_comp = predicted_aqi_72h if isinstance(predicted_aqi_72h, (int, float)) else 0

    if current_aqi_comp > 150 or (can_forecast and predict_aqi_comp > 150):
         alert_level = max(current_aqi_comp, predict_aqi_comp)
         alert_cat, _ = get_aqi_category(alert_level)
         st.error(f"**Hazardous AQI Alert:** AQI is currently or forecast to reach potentially unhealthy levels ({alert_cat}). Consider reducing outdoor activity.")
except Exception as e:
     st.warning(f"Could not evaluate alert level: {e}")


col1, col2 = st.columns(2)
with col1:
    st.metric(
        label=f"Most Recent AQI ({current_time_display})",
        value=current_aqi_display_val,
        help="This is the most recent AQI value calculated from available data."
    )
    st.markdown(f"**Level:** <span style='color:{current_col_display}; font-weight:bold;'>{current_cat_display}</span>", unsafe_allow_html=True)

with col2:
    st.metric(
        label=f"Forecast AQI (+72 Hours) ({forecast_time_display})",
        value=predicted_aqi_72h if can_forecast else "N/A",
        help="Predicted AQI value 72 hours from the last complete data point."
    )
    if can_forecast:
         st.markdown(f"**Level:** <span style='color:{forecast_col_display}; font-weight:bold;'>{forecast_cat_display}</span>", unsafe_allow_html=True)
    else:
         st.warning("Forecast could not be generated.")

# --- Simplified Chart ---
# Ensure forecast_time_utc is valid before trying to convert
if can_forecast and current_aqi_display_val != "N/A" and isinstance(predicted_aqi_72h, (int, float)) and forecast_time_utc is not pd.NaT and latest_available_time_utc is not pd.NaT:
    try:
        # Get local times for chart
        most_recent_aqi_time_local = latest_available_time_utc.tz_convert(karachi_tz)
        forecast_time_local_chart = forecast_time_utc.tz_convert(karachi_tz)

        chart_data = pd.DataFrame({
            'Time': [most_recent_aqi_time_local, forecast_time_local_chart],
            'AQI': [current_aqi, predicted_aqi_72h]
        }).set_index('Time')

        st.line_chart(chart_data, color="#FF7E00")
        st.caption("Note: Chart shows the most recent actual AQI and the predicted AQI 72 hours later.")
    except Exception as e:
         st.warning(f"Could not display forecast chart: {e}")
elif current_aqi_display_val != "N/A":
     st.info("Displaying only the most recent AQI value as forecast could not be generated.")
     # Optionally show just the current point if needed
else:
     st.warning("No current or forecast AQI data available to plot.")


# --- AQI Categories Expander ---
with st.expander("What do the AQI levels mean?"):
    st.image("https://files.airnowtech.org/airnow/today/aqi_legend.jpg", caption="Source: US EPA AirNow")

st.sidebar.header("About This Project")
st.sidebar.info(
    "This app is part of the **Pearls AQI Predictor** project. "
    "It uses a pre-trained `ExtraTreesRegressor` model (`aqi_model_extratrees.joblib`) "
    "to forecast the Air Quality Index (AQI) in Karachi 72 hours in advance. "
    "It fetches live weather and recent pollution data from Open-Meteo, "
    "engineers features, and generates a prediction based on the latest available data."
)
st.sidebar.caption("Model and data processing logic developed based on project requirements.")

