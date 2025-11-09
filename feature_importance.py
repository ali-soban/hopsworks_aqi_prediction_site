import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap  

print("Loading data and model...")
try:
    df_final = pd.read_csv('karachi_model_ready_data.csv', parse_dates=True, index_col='time')
    #gave best performance
    model = joblib.load('aqi_model_extratrees.joblib')
    
    print("Files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure 'karachi_model_ready_data.csv' and 'aqi_model_extratrees.joblib' are in the same directory.")
    exit()


TARGET_COLUMN = 'target_AQI'
X = df_final.drop(columns=[TARGET_COLUMN])
y = df_final[TARGET_COLUMN]

test_size = 0.2
split_index = int(len(df_final) * (1 - test_size))

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(f"Loaded {len(X_test)} samples for test set.")

print("\nInitializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)


if len(X_test) > 500:
    print("Calculating SHAP values for a sample of 500 test data points...")
    X_test_sample = shap.sample(X_test, 500)
else:
    print("Calculating SHAP values for the test set...")
    X_test_sample = X_test

shap_values = explainer.shap_values(X_test_sample)

print("SHAP values calculated.")

print("Generating and saving SHAP summary plot...")
try:
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    
    plt.title("SHAP Feature Importance (ExtraTrees Model)")
    plt.savefig('shap_summary_plot.png', bbox_inches='tight')
    plt.close() 
    
    print("Saved plot to 'shap_summary_plot.png'")
    
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP Feature Impact Summary (ExtraTrees Model)")
    plt.savefig('shap_beeswarm_plot.png', bbox_inches='tight')
    plt.close()
    
    print("Saved detailed plot to 'shap_beeswarm_plot.png'")
    print("\n--- SCRIPT COMPLETED ---")

except Exception as e:
    print(f"Could not create or save plot: {e}")
