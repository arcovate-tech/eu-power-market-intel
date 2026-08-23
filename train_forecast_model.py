import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_parquet("synthetic_price_data.parquet")
df = df.sort_values("datetime").reset_index(drop=True)

# --- Feature engineering ---
df["lag_1h"] = df["price_eur_mwh"].shift(1)
df["lag_24h"] = df["price_eur_mwh"].shift(24)      # same hour yesterday
df["lag_168h"] = df["price_eur_mwh"].shift(168)     # same hour last week

df = df.dropna().reset_index(drop=True)

feature_cols = [
    "hour", "day_of_week", "wind_pct", "solar_pct", "gas_pct", "nuclear_pct",
    "lag_1h", "lag_24h", "lag_168h"
]
target_col = "price_eur_mwh"

# --- Time-based train/test split (last 14 days = test) ---
split_point = df["datetime"].max() - pd.Timedelta(days=14)
train = df[df["datetime"] <= split_point]
test = df[df["datetime"] > split_point]

X_train, y_train = train[feature_cols], train[target_col]
X_test, y_test = test[feature_cols], test[target_col]

# --- Baseline: naive "yesterday's price" ---
baseline_pred = test["lag_24h"]
baseline_mape = mean_absolute_percentage_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

# --- LightGBM model ---
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

lgb_pred = model.predict(X_test)
lgb_mape = mean_absolute_percentage_error(y_test, lgb_pred)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))

# --- Report ---
print("=== Baseline (naive yesterday's price) ===")
print(f"MAPE: {baseline_mape:.2%}  RMSE: {baseline_rmse:.2f}")

print("\n=== LightGBM model ===")
print(f"MAPE: {lgb_mape:.2%}  RMSE: {lgb_rmse:.2f}")

# --- Feature importance ---
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Feature importance ===")
print(importance)

# --- Plot actual vs predicted ---
plt.figure(figsize=(14, 5))
plt.plot(test["datetime"], y_test.values, label="Actual", linewidth=1.5)
plt.plot(test["datetime"], lgb_pred, label="Predicted (LightGBM)", linewidth=1.5)
plt.title("Actual vs Predicted Day-Ahead Price")
plt.xlabel("Date")
plt.ylabel("Price (EUR/MWh)")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_actual_vs_predicted.png")
print("\nPlot saved as forecast_actual_vs_predicted.png")