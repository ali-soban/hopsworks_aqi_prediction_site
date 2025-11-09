import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading 'karachi_model_ready_data.csv'...")
try:
    df_final = pd.read_csv('karachi_model_ready_data.csv', parse_dates=True, index_col='time')
except FileNotFoundError:
    print("Error: 'karachi_model_ready_data.csv' not found.")
    print("Please make sure the feature-engineered file is in the same directory.")
    exit()

print("File loaded")

print("\n--- DATA PREPARATION ---")

TARGET_COLUMN = 'target_AQI'
X = df_final.drop(columns=[TARGET_COLUMN])
y = df_final[TARGET_COLUMN]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

test_size = 0.2
split_index = int(len(df_final) * (1 - test_size))

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

common_params = {
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1
}

models = {
    "RandomForest": RandomForestRegressor(
        **common_params, 
        max_depth=35, 
        min_samples_leaf=10
    ),
    "ExtraTrees": ExtraTreesRegressor(
        **common_params, 
        max_depth=35, 
        min_samples_leaf=10
    ),
    "XGBoost": XGBRegressor(
        **common_params, 
        max_depth=20, 
        learning_rate=0.1
    ),
    "LightGBM": LGBMRegressor(
        **common_params, 
        learning_rate=0.1, 
        num_leaves=41,
        verbose=-1
    )
}

results_list = []
predictions_df = pd.DataFrame({'actual_AQI': y_test}, index=y_test.index)

for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, y_train)
    print(f"Training complete.")
    try:
        joblib.dump(model, f'aqi_model_{model_name.lower()}.joblib')
        print(f"Model saved as 'aqi_model_{model_name.lower()}.joblib'")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    print(f"Making predictions with {model_name}...")
    y_pred = model.predict(X_test)
    predictions_df[f'predicted_{model_name}'] = y_pred
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Evaluation Metrics for {model_name} ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):    {mae:.4f}")
    print(f"R-squared (R²):               {r2:.4f}")
    
    results_list.append({
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

print("\n--- MODEL COMPARISON ---")
metrics_df = pd.DataFrame(results_list).set_index("Model")
print(metrics_df.sort_values(by="RMSE"))

print("\nSaving test results and plotting...")

try:
    predictions_df.to_csv('karachi_model_predictions_all.csv')
    print("Saved all predictions to 'karachi_model_predictions_all.csv'")
except Exception as e:
    print(f"Could not save predictions CSV: {e}")





n_samples = min(200, len(predictions_df))
plot_df = predictions_df.sample(n=n_samples, random_state=42, replace=False).sort_index()

try:
    plt.figure(figsize=(15, 8))
    
    plt.plot(plot_df.index, plot_df['actual_AQI'], label='Actual AQI', alpha=0.9, 
             marker='o', linestyle='None', markersize=6, color='black')
    
    colors = ['blue', 'green', 'red', 'purple']
    for i, model_name in enumerate(models.keys()):
        plt.plot(plot_df.index, plot_df[f'predicted_{model_name}'], 
                 label=f'Predicted ({model_name})', linestyle='--', 
                 alpha=0.7, color=colors[i])

    plt.title(f'Model Comparison: Actual vs. Predicted AQI (72-Hour Forecast)\n(Plotting a random sample of {n_samples} test points)')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig('model_comparison_plot.png')
    print("Saved model comparison plot to 'model_comparison_plot.png'")
    print("\n--- SCRIPT COMPLETED ---")

except Exception as e:
    print(f"Could not create or save plot: {e}")
