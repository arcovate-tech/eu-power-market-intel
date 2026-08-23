import pandas as pd
import numpy as np

np.random.seed(42)

# 90 days of hourly data
start = pd.Timestamp("2026-05-01")
hours = pd.date_range(start=start, periods=90*24, freq="h")

df = pd.DataFrame({"datetime": hours})
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek

# Fake generation mix (%) - wind varies randomly, rest fills the gap
df["wind_pct"] = np.clip(np.random.normal(30, 15, len(df)), 0, 90)
df["solar_pct"] = np.clip(
    np.sin((df["hour"] - 6) / 24 * 2 * np.pi) * 20 + 10, 0, 60
) * (df["hour"].between(6, 18)).astype(int)
df["gas_pct"] = np.clip(100 - df["wind_pct"] - df["solar_pct"] - 20, 5, 80)
df["nuclear_pct"] = 100 - df["wind_pct"] - df["solar_pct"] - df["gas_pct"]

# Fake price - higher during peak hours, lower when wind/solar is high, plus noise
base_price = 60
peak_bump = df["hour"].apply(lambda h: 25 if 7 <= h <= 21 else -10)
renewable_effect = -0.4 * (df["wind_pct"] + df["solar_pct"])
weekend_effect = df["day_of_week"].apply(lambda d: -8 if d >= 5 else 0)
noise = np.random.normal(0, 8, len(df))

df["price_eur_mwh"] = base_price + peak_bump + renewable_effect + weekend_effect + noise
df["price_eur_mwh"] = df["price_eur_mwh"].round(2)

df.to_parquet("synthetic_price_data.parquet", index=False)
print(f"Generated {len(df)} rows.")
print(df.head())