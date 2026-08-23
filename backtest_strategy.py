import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_parquet("synthetic_price_data.parquet")
df = df.sort_values("datetime").reset_index(drop=True)

df["lag_1h"] = df["price_eur_mwh"].shift(1)
df["lag_24h"] = df["price_eur_mwh"].shift(24)
df["lag_168h"] = df["price_eur_mwh"].shift(168)
df = df.dropna().reset_index(drop=True)
df["date"] = df["datetime"].dt.date

feature_cols = ["hour", "day_of_week", "wind_pct", "solar_pct", "gas_pct", "nuclear_pct",
                "lag_1h", "lag_24h", "lag_168h"]

split_point = df["datetime"].max() - pd.Timedelta(days=14)
train = df[df["datetime"] <= split_point]
test = df[df["datetime"] > split_point].copy()

model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(train[feature_cols], train["price_eur_mwh"])
test["predicted_price"] = model.predict(test[feature_cols])

# For each test day: decide buy/sell hour using PREDICTED price, execute at ACTUAL price
results = []
for date, group in test.groupby("date"):
    if len(group) < 24:
        continue
    buy_hour_row = group.loc[group["predicted_price"].idxmin()]
    sell_hour_row = group.loc[group["predicted_price"].idxmax()]
    actual_buy_price = buy_hour_row["price_eur_mwh"]
    actual_sell_price = sell_hour_row["price_eur_mwh"]
    pnl = actual_sell_price - actual_buy_price
    results.append({"date": date, "pnl": pnl})

backtest_df = pd.DataFrame(results)
total_pnl = backtest_df["pnl"].sum()
win_rate = (backtest_df["pnl"] > 0).mean()
cumulative_pnl = backtest_df["pnl"].cumsum()
max_drawdown = (cumulative_pnl - cumulative_pnl.cummax()).min()

print("=== Forecast-Driven Storage Arbitrage Backtest ===")
print(f"Trading days: {len(backtest_df)}")
print(f"Total PnL: €{total_pnl:.2f} per MWh cycled")
print(f"Win rate: {win_rate:.1%}")
print(f"Max drawdown: €{max_drawdown:.2f}")

backtest_df.to_csv("backtest_results.csv", index=False)