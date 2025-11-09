# 10 Pearls AQI Predictor: Final Project Report

**Author:** Ali Soban (aalliisn72@gmail.com)
**Project:** End-to-end MLOps pipeline for 3-day AQI forecasting in Karachi.

## 1. Project Overview

This report documents the design, implementation, and deployment of a 100% serverless, end-to-end MLOps pipeline to predict the Air Quality Index (AQI) in Karachi, Pakistan, 72 hours in advance.

### Final Deliverables Achieved
1.  **End-to-end AQI prediction system:** A live, functioning system.
2.  **Scalable, automated pipeline:** GitHub Actions and Hopsworks work together to automatically update features and retrain models.
3.  **Interactive dashboard:** A Streamlit application provides real-time AQI and 3-day forecasts to the user.
4.  **Detailed report:** This document.


## 2. System Architecture

* **Data Producer (Hourly):** A GitHub Actions workflow runs hourly. It fetches raw weather and pollution data from the Open-Meteo API, engineers a comprehensive set of features, and inserts this data into a Hopsworks Feature Group.
* **Model Trainer (Daily):** A separate GitHub Actions workflow runs daily. It connects to the Hopsworks Feature Store, reads all the latest feature data, trains a new `ExtraTreesRegressor` model, and publishes the validated model to the Hopsworks Model Registry.
* **Inference Consumer (Live):** A Streamlit web application connects directly to Hopsworks. It pulls the *latest-trained model* from the Model Registry and the *latest feature vector* from the Feature Store to generate and display a live 72-hour forecast for the user.

### Technology Stack
* **Data Source:** Open-Meteo (Weather and Air Quality APIs)
* **Feature Store:** Hopsworks
* **Model Registry:** Hopsworks
* **Automation (CI/CD):** GitHub Actions
* **Data Processing:** Pandas, NumPy
* **Model Training:** Scikit-learn
* **Model Analysis:** SHAP
* **Web Application:** Streamlit

## 3. Data Processing & Feature Engineering

### 3.1. Data Source
We used the **Open-Meteo API** for its free, high-quality, and non-API-key access to both historical weather and raw air pollutant data.

### 3.2. Target Variable (AQI)
The API provides raw pollutant concentrations. We implemented a Python function based on the **US EPA's official standard** to get the target AQI value.

### 3.3. Feature Engineering
To prepare the data for a time-series model, we engineered three main categories of features, as required by the project guidelines:

1.  **Time-Based (Cyclical) Features:** To represent the time of day, day of week, and month of year, we used sine and cosine transformations.
2.  **Lagged Features (Momentum):** The model was fed historical data, such as `aqi_lag_1hr`, `aqi_lag_24hr`..
3.  **Rolling Features (Trend):** We included a `aqi_rolling_3hr` (3-hour rolling average) to smooth out short-term noise.


## 4. Model Training & Evaluation

### 4.1. Model Selection
We experimented with several tree-based models that do not require feature scaling. The models included `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, and `ExtraTreesRegressor`.

After comparison, the **`ExtraTreesRegressor`** provided the best performance (lowest error) and was selected as our production model.

### 4.2. Feature Importance (SHAP Analysis)
We used SHAP, as per the guidelines, to explain the model's predictions. The analysis revealed that **`month_cos`** was the single most important feature.

This is a critical insight: The model learned that the time of year (e.g., winter vs. summer) is more predictive of the 3-day forecast than even the current day's weather or AQI.

## 5. Automation (CI/CD Pipeline)

We successfully implemented a fully automated pipeline using GitHub Actions.

* **`hourly_feature_pipeline.yml`:** This "Producer" workflow runs hourly. It executes a Python script that fetches, engineers, and inserts the latest feature data into the `karachi_aqi_features` Feature Group in Hopsworks. We used `get_or_create_feature_group` to ensure the schema is always correct, instead of mannually creating the feature group on Hopsworks, as that caused errors as well.
* **`daily_model_training.yml`:** This "Trainer" workflow runs daily. It queries the Feature Group, trains the `ExtraTrees` model on all available data, and saves the new model (with its schema and metrics) to the Hopsworks Model Registry as `karachi_aqi_predictor`.

## 6. Interactive Dashboard (Streamlit)

The final deliverable is an interactive dashboard built with Streamlit.

* **Connection:** The app connects securely to Hopsworks using Streamlit Secrets.
* **Inference:** On load, it fetches the *latest* model from the Model Registry and the *latest* feature vector* from the Feature Store.
* **Display:** It displays:
    1.  The **Most Recent AQI** value.