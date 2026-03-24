# UK Weather-Driven Gas Demand Forecast

A machine learning model that forecasts daily UK gas demand from weather data, with walk-forward validation on Gas Year boundaries and baseline comparisons.

**[Live Dashboard](https://arkady-zelman-uk-gas-demand-forecast-srcdashboardapp-8liq15.streamlit.app/)** — no setup required, runs in the browser.

## Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run src/dashboard/app.py
```

Everything is fetched automatically on first load.

## Approach

### Data

| Source | What | API | Auth |
|---|---|---|---|
| National Gas | NTS daily demand (mcm/d), NDM/DM by LDZ | `data.nationalgas.com` | None |
| Open-Meteo | Historical + 16-day forecast weather | `open-meteo.com` | None |

Weather is population-weighted across 5 UK cities (London 40%, Birmingham 18%, Manchester 15%, Edinburgh 15%, Cardiff 12%) to approximate the national heating load.

### Features

The feature set reflects domain knowledge about what drives gas demand:

- **HDD / CDD**: Heating and cooling degree days — the primary demand signal
- **Wind chill**: Effective temperature using JAG/TI formula; amplifies heating demand on cold windy days
- **Autoregressive lags**: demand_lag_1, demand_lag_7, temperature lags (1-3 days), HDD lag
- **Rolling context**: 7-day moving averages of demand and temperature
- **Calendar**: day-of-week, month, weekend flag, UK bank holidays
- **Gas year structure**: quarter (Q1-Q4), winter flag, long-run trend
- **COVID regime**: Binary flag for lockdown period (Mar 2020 – Jun 2021) when commercial/industrial demand was ~15-20% below normal

### Models

**XGBoost** is the production model.  We also run two baselines through the same walk-forward folds to quantify the lift:

| Model | Role |
|---|---|
| XGBoost | Production |
| Seasonal naive | Baseline |
| Linear | Baseline |

An **LSTM** module is included for future development.

### Validation

Walk-forward backtesting on Gas Year boundaries (expanding window):

| Fold | Training | Test |
|---|---|---|
| 1 | GY 2020/21 – GY 2021/22 | GY 2022/23 |
| 2 | GY 2020/21 – GY 2022/23 | GY 2023/24 |
| 3 | GY 2020/21 – GY 2023/24 | GY 2024/25 |

Each test set is a full gas year the model never saw during training — no data leakage.

**Important caveat:** this is a **1-step-ahead** backtest.  Demand lags use actual prior-day values, which reflects the operational setting where a trader has yesterday's outturn each morning.  The 16-day recursive forecast on the dashboard is a different object — prediction error compounds because each day's lags come from prior predictions, not actuals.  Expect day-7+ accuracy to be worse than the backtest headline implies.

### Metrics & Diagnostics

We lead with **RMSE** and **MAE** because they're stable at all demand levels.  MAPE is reported for intuition but should be treated with caution — it blows up on low-demand summer days (5 mcm error on a 140 mcm/d day looks much worse in percentage terms than the same error on a 300 mcm/d winter day).

Diagnostics include:
- **SHAP partial dependence** for the top 3 features — shows _how_ each driver affects the prediction, not just that it matters
- **Residual ACF** — checks whether errors are autocorrelated (would indicate missing temporal structure)
- **Residuals vs fitted**, over time, and distribution histogram

### Uncertainty

Prediction intervals use **empirical quantiles** (5th/95th percentile) from the walk-forward residuals rather than assuming Gaussian errors.  The residual distribution is typically right-skewed in winter due to cold snap surprises, so parametric intervals would be miscalibrated.

### Structural Breaks

The COVID lockdown period (Mar 2020 – Jun 2021) is flagged as a regime dummy.  Without it, the model attributes the demand drop to weather and learns a spurious cold-sensitivity bias.  The `days_since_start` trend captures longer-run efficiency gains in building stock and boiler technology — it is anchored to a fixed epoch date (config `date_range.start`) so that the feature value is identical in training and live inference.

## Dashboard

| Page | Content |
|---|---|
| **Overview** | Headline metrics, baseline lift, 14-day forecast with temperature context |
| **Historical Fit** | Actual vs predicted (OOS default), residuals, date range selector |
| **Model Diagnostics** | Feature importance, SHAP dependence, ACF, walk-forward fold table, residual plots |
| **Weather Impact** | Temperature/HDD/wind vs demand scatters, monthly seasonality |
| **Forecast** | Live 16-day forecast with empirical prediction intervals |

## Project Structure

```
config/settings.yaml        — hyperparameters, weather cities, date ranges
data/cache/                  — parquet cache (auto-generated, gitignored)
src/
  config.py                  — YAML config loader
  data/
    demand_client.py         — National Gas API client
    weather_client.py        — Open-Meteo API client
    cache.py                 — parquet cache layer
  features/
    engineer.py              — 24-feature pipeline (HDD, lags, calendar, COVID)
    holidays.py              — UK bank holidays
  model/
    xgboost_model.py         — production XGBoost model
    lstm_model.py            — LSTM comparison baseline
    baselines.py             — seasonal naive + linear regression
    backtest.py              — walk-forward engine (runs XGBoost + baselines)
    evaluate.py              — metrics, residual ACF
    forecast.py              — live forecast with empirical intervals
  dashboard/
    app.py                   — Streamlit dashboard
notebooks/
  exploration.ipynb          — EDA notebook
tests/                       — pytest suite (30 tests)
```

## Gas Year Conventions

UK gas year runs October to September:
- **Q1** (Oct-Dec) and **Q2** (Jan-Mar) = winter, peak demand
- **Q3** (Apr-Jun) and **Q4** (Jul-Sep) = summer, baseload

## Running Tests

```bash
pytest tests/ -v
```

## Dependencies

Core: `xgboost`, `scikit-learn`, `pandas`, `numpy`, `streamlit`, `plotly`, `shap`, `statsmodels`

Data: `requests`, `pyarrow`, `pyyaml`, `holidays`

Optional: `torch` — only needed for the LSTM comparison baseline, not the dashboard or XGBoost model.  Install manually with `pip install torch` if you want to run the LSTM experiments.
