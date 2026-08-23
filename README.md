# European Day-Ahead Power & Gas Market Intelligence System

An end-to-end pipeline that forecasts European day-ahead electricity prices, extracts structured trading signals from live energy news using an LLM agent, and backtests a simple storage-arbitrage strategy — presented in a live daily market brief dashboard.

**Live dashboard**: https://eu-power-market-intel-d65twmdrztrgigvuj2dgfa.streamlit.app/

## What this does

1. **Forecasting** — LightGBM model predicts next-day hourly electricity prices using lag features, generation mix, and calendar features. Beats a naive "yesterday's price" baseline (19.9% MAPE vs 33.6% baseline).
2. **News-to-signal agent** — LangChain + Groq (LLM) pipeline pulls live energy market news and extracts structured JSON signals: `{commodity, sentiment, key_driver, confidence}`.
3. **Backtest** — Forecast-driven storage arbitrage strategy: predicts the day's cheapest/most expensive hours in advance, evaluates PnL against realized prices (no look-ahead bias).
4. **Dashboard** — Streamlit app combining all three into a single market brief view.

## Data sources

- **Price & generation mix data**: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) — the official EU electricity/gas market data source, via the `entsoe-py` Python client. *(API access currently pending approval — pipeline is currently demonstrated on synthetic data shaped identically to real ENTSO-E output; swap-in is a one-line change once the key is approved.)*
- **News signals**: Live RSS feed (Google News, energy market search) — no API key required.

## Data definitions

- **Bidding zone**: A geographic area within which electricity trades at a single wholesale price (e.g. Germany-Luxembourg). Prices can differ across zones due to transmission constraints.
- **Day-ahead price**: The price at which electricity for a given hour tomorrow is traded today, in EUR/MWh.
- **Generation mix %**: Share of total generation from each source (wind, solar, gas, nuclear) for a given hour.

## Methodology

- **Forecasting**: LightGBM regression, trained on lag features (1h, 24h, 168h), hour-of-day, day-of-week, and generation mix. Evaluated with a **time-based** train/test split (last 14 days held out) — not random — since random splits leak future information into time-series models.
- **News agent**: LangChain prompt template + Groq LLM (`openai/gpt-oss-120b`), constrained to return structured JSON only.
- **Backtest**: For each day, the model predicts which hour will be cheapest and which will be most expensive; PnL is calculated using the *actual* realized prices at those predicted hours (not the true daily min/max), avoiding look-ahead bias.

## Known limitations

- Currently running on synthetic data pending ENTSO-E API approval — real-data results will differ (likely more realistic/varied) once live data is integrated.
- Single bidding zone (Germany-Luxembourg) — designed to extend to France/Netherlands.
- Backtest is illustrative only, not investment advice.
- No weather features yet (planned addition).
- Cloud-scheduled auto-refresh (AWS Lambda/EventBridge) not yet implemented — dashboard currently runs computations on-demand.

## Tech stack

Python, `entsoe-py`, Pandas, LightGBM, scikit-learn, LangChain, Groq API, Streamlit, Plotly, R (EDA), GitHub, Streamlit Community Cloud.

## Reproducing this

```bash
git clone https://github.com/arcovate-tech/eu-power-market-intel.git
cd eu-power-market-intel
pip install -r requirements.txt
python generate_synthetic_data.py
python train_forecast_model.py
python backtest_strategy.py
streamlit run dashboard.py
```

## Author

Aman singh — built as a portfolio project targeting energy trading data roles.