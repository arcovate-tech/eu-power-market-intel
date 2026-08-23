import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import json
import subprocess
import sys
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="European Power Market Brief", layout="wide")
st.title("⚡ European Day-Ahead Power Market Brief")
st.caption("Demo build — currently running on synthetic data pending ENTSO-E API approval")

# ============ PANEL 1: FORECAST ============
st.header("📈 Next-Day Price Forecast")

import boto3
import io

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
bucket = os.getenv("AWS_BUCKET_NAME")

obj = s3.get_object(Bucket=bucket, Key="data/synthetic_price_data.parquet")
df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
df = df.sort_values("datetime").reset_index(drop=True)
df["lag_1h"] = df["price_eur_mwh"].shift(1)
df["lag_24h"] = df["price_eur_mwh"].shift(24)
df["lag_168h"] = df["price_eur_mwh"].shift(168)
df = df.dropna().reset_index(drop=True)

feature_cols = ["hour", "day_of_week", "wind_pct", "solar_pct", "gas_pct", "nuclear_pct",
                "lag_1h", "lag_24h", "lag_168h"]

split_point = df["datetime"].max() - pd.Timedelta(days=14)
train = df[df["datetime"] <= split_point]
test = df[df["datetime"] > split_point].copy()

model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
model.fit(train[feature_cols], train["price_eur_mwh"])
test["predicted_price"] = model.predict(test[feature_cols])

mape = mean_absolute_percentage_error(test["price_eur_mwh"], test["predicted_price"])
rmse = np.sqrt(mean_squared_error(test["price_eur_mwh"], test["predicted_price"]))

col1, col2 = st.columns(2)
col1.metric("Model MAPE", f"{mape:.1%}")
col2.metric("Model RMSE", f"€{rmse:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=test["datetime"], y=test["price_eur_mwh"], name="Actual", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=test["datetime"], y=test["predicted_price"], name="Predicted", line=dict(color="#ff7f0e")))
fig.update_layout(xaxis_title="Date", yaxis_title="Price (EUR/MWh)", height=400)
st.plotly_chart(fig, width='stretch')

# ============ PANEL 2: NEWS SIGNALS ============
st.header("📰 Today's Market Signals")

try:
    result = subprocess.run([sys.executable, "news_agent.py"], capture_output=True, text=True, timeout=60)
    st.text(result.stdout[-2000:] if result.stdout else "No output captured.")
except Exception as e:
    st.warning(f"Could not run news agent live: {e}")

# ============ PANEL 3: BACKTEST ============
st.header("🔋 Backtest: Forecast-Driven Storage Arbitrage")

backtest_df = pd.read_csv("backtest_results.csv")
total_pnl = backtest_df["pnl"].sum()
win_rate = (backtest_df["pnl"] > 0).mean()
cumulative = backtest_df["pnl"].cumsum()
max_dd = (cumulative - cumulative.cummax()).min()

c1, c2, c3 = st.columns(3)
c1.metric("Total PnL", f"€{total_pnl:.2f} / MWh")
c2.metric("Win Rate", f"{win_rate:.0%}")
c3.metric("Max Drawdown", f"€{max_dd:.2f}")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=backtest_df["date"], y=cumulative, name="Cumulative PnL", line=dict(color="#2ca02c")))
fig2.update_layout(xaxis_title="Date", yaxis_title="Cumulative PnL (EUR/MWh)", height=300)
st.plotly_chart(fig2, width='stretch')

st.caption("⚠️ Illustrative backtest only — not investment advice. Currently on synthetic data.")