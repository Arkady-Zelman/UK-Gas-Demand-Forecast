"""Streamlit dashboard — UK Gas Demand Forecast.

python -m streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.demand_client import DemandClient
from src.data.weather_client import WeatherClient
from src.data import cache
from src.features.engineer import build_features, FEATURE_COLS
from src.model.xgboost_model import GasDemandXGB
from src.model.backtest import walk_forward_xgb, BacktestResult
from src.model.evaluate import compute_metrics, residual_acf
from src.model.forecast import generate_forecast
from src.config import get

# -- page config -------------------------------------------------------

st.set_page_config(
    page_title="UK Gas Demand Forecast",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- styling -----------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-card: rgba(255,255,255,0.03);
    --bg-card-hover: rgba(255,255,255,0.05);
    --border: rgba(255,255,255,0.06);
    --border-warm: rgba(212,132,94,0.25);
    --muted: #6b7080;
    --accent: #d4845e;
    --accent-bg: rgba(212,132,94,0.12);
    --r-lg: 20px;
    --r-md: 14px;
    --r-sm: 10px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
section.main .block-container {
    animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) both;
    max-width: 1200px;
    padding-top: 2rem;
}

div[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 12px;
    background: var(--bg-card);
    margin-bottom: 1.2rem;
    transition: border-color 0.3s ease;
}
div[data-testid="stPlotlyChart"]:hover { border-color: var(--border-warm); }

div[data-testid="stMetric"] {
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 16px 20px;
    background: var(--bg-card);
    transition: border-color 0.25s ease, transform 0.15s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--border-warm);
    transform: translateY(-1px);
}
div[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-weight: 400 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 500 !important;
    font-size: 1.6rem !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    overflow: hidden;
    background: var(--bg-card);
}

section[data-testid="stSidebar"] {
    background: #0c0d0f !important;
    border-right: 1px solid var(--border);
}
div[data-testid="stSidebar"] .stRadio > div { gap: 0.15rem; }
div[data-testid="stSidebar"] .stRadio label {
    border-radius: var(--r-sm) !important;
    padding: 0.45rem 0.8rem !important;
    transition: background 0.2s ease;
}
div[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.04); }

.stButton > button {
    border-radius: var(--r-sm) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    transition: all 0.2s ease !important;
    font-weight: 400 !important;
}
.stButton > button:hover {
    border-color: var(--border-warm) !important;
    background: var(--bg-card-hover) !important;
}

h1, h2, h3 { font-weight: 500 !important; letter-spacing: -0.02em; }
h1 { font-size: 1.7rem !important; }
h2 { font-size: 1.3rem !important; }
h3 { font-size: 1.1rem !important; color: var(--muted) !important; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

div[data-testid="stAlert"] {
    border-radius: var(--r-md) !important;
    border: 1px solid var(--border) !important;
}
details {
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    background: var(--bg-card) !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# -- shared chart config ------------------------------------------------

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c8cad0", family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
)

WARM = "#d4845e"
COOL = "#5b9bd5"
GREEN = "rgba(72,199,142,0.6)"
RED = "rgba(231,106,100,0.6)"


# -- data loaders (always run — fast) -----------------------------------

@st.cache_data(ttl=3600, show_spinner="Fetching demand data...")
def load_demand() -> pd.DataFrame | None:
    cached = cache.load("nts_demand", max_age_hours=24)
    if cached is not None and not cached.empty:
        return cached
    df = DemandClient().get_all_demand(start=get("date_range.start", "2020-10-01"))
    if df is not None and not df.empty:
        cache.save("nts_demand", df)
    return df


@st.cache_data(ttl=3600, show_spinner="Fetching weather data...")
def load_weather() -> pd.DataFrame | None:
    cached = cache.load("weather_historical", max_age_hours=24)
    if cached is not None and not cached.empty:
        return cached
    df = WeatherClient().get_historical(start=get("date_range.start", "2020-10-01"))
    if df is not None and not df.empty:
        cache.save("weather_historical", df)
    return df


@st.cache_resource(show_spinner="Training model...")
def train_model(_h: str) -> GasDemandXGB | None:
    demand, weather = load_demand(), load_weather()
    if demand is None or weather is None:
        return None
    df = build_features(demand, weather)
    if df is None or df.empty:
        return None
    m = GasDemandXGB()
    m.fit(df)
    return m


def _hash() -> str:
    d, w = load_demand(), load_weather()
    return f"{len(d) if d is not None else 0}_{len(w) if w is not None else 0}"


def _features():
    demand, weather = load_demand(), load_weather()
    if demand is None or weather is None:
        return None
    return build_features(demand, weather)


# -- backtest (expensive — only on demand) ------------------------------

@st.cache_data(ttl=7200, show_spinner="Running walk-forward backtest...")
def _run_backtest_cached(_h: str) -> BacktestResult | None:
    demand, weather = load_demand(), load_weather()
    if demand is None or weather is None:
        return None
    df = build_features(demand, weather)
    if df is None or df.empty:
        return None
    try:
        return walk_forward_xgb(df)
    except Exception as e:
        st.warning(f"Backtest failed: {e}")
        return None


def _get_backtest() -> BacktestResult | None:
    """Return backtest result if it has been run, else None."""
    return st.session_state.get("backtest_result")


def _trigger_backtest():
    """Run backtest and store in session state."""
    h = _hash()
    result = _run_backtest_cached(h)
    st.session_state["backtest_result"] = result


def _backtest_button(location: str = "main") -> bool:
    """Show a 'Run Full Analysis' button.  Returns True if backtest is available."""
    bt = _get_backtest()
    if bt is not None:
        return True
    st.info("Full analysis (walk-forward backtest, baselines, SHAP) has not been run yet.")
    if st.button("Run Full Analysis", key=f"bt_{location}", use_container_width=True):
        _trigger_backtest()
        st.rerun()
    return False


# -- sidebar ------------------------------------------------------------

with st.sidebar:
    st.markdown("### UK Gas Demand Forecast")
    st.caption("Weather-driven XGBoost model")

    page = st.radio(
        "Page", ["Overview", "Historical Fit", "Model Diagnostics",
                 "Weather Impact", "Forecast"],
        index=0, label_visibility="collapsed",
    )
    st.divider()

    if _get_backtest() is None:
        if st.button("Run Full Analysis", key="sidebar_bt", use_container_width=True):
            _trigger_backtest()
            st.rerun()
        st.caption("Runs walk-forward backtest, baselines, and SHAP")
    else:
        st.caption("Full analysis complete")

    st.divider()
    if st.button("Refresh Data", use_container_width=True):
        cache.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pop("backtest_result", None)
        st.rerun()
    st.caption(f"Data from {get('date_range.start')} onward")
    st.caption("National Gas · Open-Meteo")


# == OVERVIEW ===========================================================

def page_overview():
    st.header("Overview")

    h = _hash()
    model = train_model(h)
    if model is None:
        st.error("Could not load data or train model.")
        return

    bt = _get_backtest()

    # If backtest available, show out-of-sample metrics
    if bt is not None:
        mt = bt.metrics_table
        mean = mt[mt["fold"] == "Mean"].iloc[0]

        st.caption("WALK-FORWARD OUT-OF-SAMPLE PERFORMANCE")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{mean['rmse_mcm']:.1f} mcm/d")
        c2.metric("MAPE", f"{mean['mape_pct']:.1f}%")
        c3.metric("R²", f"{mean['r2']:.3f}")
        c4.metric("MAE", f"{mean['mae_mcm']:.1f} mcm/d")

        bc = bt.baseline_comparison
        if not bc.empty:
            bc_mean = bc[bc["fold"] == "Mean"]
            if not bc_mean.empty:
                row = bc_mean.iloc[0]
                naive_rmse = row.get("seasonal_naive_rmse", 0)
                linear_rmse = row.get("linear_rmse", 0)
                xgb_rmse = row.get("xgboost_rmse", 0)
                if naive_rmse > 0:
                    lift_naive = (1 - xgb_rmse / naive_rmse) * 100
                    lift_linear = (1 - xgb_rmse / linear_rmse) * 100 if linear_rmse > 0 else 0
                    st.caption(
                        f"XGBoost reduces RMSE by **{lift_naive:.0f}%** vs seasonal naive "
                        f"and **{lift_linear:.0f}%** vs linear regression."
                    )
    else:
        # Quick in-sample metrics as a fast preview
        df = _features()
        if df is not None:
            preds = model.predict(df)
            m = compute_metrics(df["demand_mcm"].values, preds)
            st.caption("IN-SAMPLE FIT (run full analysis for out-of-sample metrics)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{m['rmse_mcm']:.1f} mcm/d")
            c2.metric("MAPE", f"{m['mape_pct']:.1f}%")
            c3.metric("R²", f"{m['r2']:.3f}")
            c4.metric("MAE", f"{m['mae_mcm']:.1f} mcm/d")

    st.markdown("")

    # Forecast — uses backtest residuals for intervals if available, else ±10%
    resid = None
    if bt is not None:
        resid = bt.all_predictions["actual"].values - bt.all_predictions["predicted"].values

    fc = generate_forecast(model, backtest_residuals=resid)

    if fc is not None and not fc.empty:
        st.subheader("14-Day Demand Forecast")
        fig = go.Figure()
        if "lower_mcm" in fc.columns:
            fig.add_trace(go.Scatter(
                x=fc["date"], y=fc["upper_mcm"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=fc["date"], y=fc["lower_mcm"], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(212,132,94,0.1)",
                name="90% interval" if bt else "±10% band"))
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["forecast_mcm"], mode="lines+markers",
            name="Demand forecast", line=dict(color=WARM, width=2.5),
            marker=dict(size=5)))
        fig.update_layout(**_LAYOUT, yaxis_title="Demand (mcm/d)", height=380,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        if "temp_mean" in fc.columns:
            st.subheader("Temperature Outlook")
            fig_t = go.Figure()
            if "temp_min" in fc.columns and "temp_max" in fc.columns:
                fig_t.add_trace(go.Scatter(x=fc["date"], y=fc["temp_max"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig_t.add_trace(go.Scatter(x=fc["date"], y=fc["temp_min"], mode="lines",
                    line=dict(width=0), fill="tonexty", fillcolor="rgba(91,155,213,0.1)",
                    name="Min–Max range"))
            fig_t.add_trace(go.Scatter(x=fc["date"], y=fc["temp_mean"],
                mode="lines+markers", name="Mean temp",
                line=dict(color=COOL, width=2.5), marker=dict(size=5)))
            fig_t.add_hline(y=15.5, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                annotation_text="HDD base (15.5°C)",
                annotation_font_color="rgba(255,255,255,0.3)",
                annotation_position="top left")
            fig_t.update_layout(**_LAYOUT, yaxis_title="Temperature (°C)", height=280,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Forecast unavailable — weather API may be down.")

    # Baseline comparison (only if backtest ran)
    if bt is not None:
        bc = bt.baseline_comparison
        with st.expander("Baseline Comparison"):
            if not bc.empty:
                st.dataframe(bc.style.format({
                    "xgboost_rmse": "{:.1f}", "xgboost_mape": "{:.1f}",
                    "seasonal_naive_rmse": "{:.1f}", "seasonal_naive_mape": "{:.1f}",
                    "linear_rmse": "{:.1f}", "linear_mape": "{:.1f}",
                }, na_rep="—"), use_container_width=True)
                st.caption(
                    "Seasonal naive = same calendar day from last year.  "
                    "Linear = Ridge regression on identical features.  "
                    "Both use the same walk-forward folds as XGBoost."
                )


# == HISTORICAL FIT =====================================================

def page_historical_fit():
    st.header("Historical Fit")

    h = _hash()
    model = train_model(h)
    bt = _get_backtest()
    demand, weather = load_demand(), load_weather()

    if demand is None or weather is None or model is None:
        st.error("Data not available.")
        return

    df = build_features(demand, weather)
    if df is None or df.empty:
        return

    # If backtest available, offer both views; otherwise just show in-sample
    if bt is not None:
        view = st.radio("View",
            ["Out-of-sample (walk-forward)", "In-sample (training set)"],
            index=0, horizontal=True)
    else:
        view = "In-sample (training set)"

    if view.startswith("Out-of-sample") and bt is not None:
        st.caption(
            "Each point is a 1-step-ahead prediction: the model never saw this gas year "
            "during training, but demand lags use actual prior-day values (as a trader "
            "would have each morning). This is not the same as the recursive multi-day "
            "forecast shown on the Forecast page, where prediction error compounds.")
        ap = bt.all_predictions.copy()
        ap["date"] = pd.to_datetime(ap["date"])

        c1, c2 = st.columns(2)
        sd = c1.date_input("Start", value=ap["date"].min().date(),
                           min_value=ap["date"].min().date(), max_value=ap["date"].max().date())
        ed = c2.date_input("End", value=ap["date"].max().date(),
                           min_value=ap["date"].min().date(), max_value=ap["date"].max().date())

        filt = ap[(ap["date"].dt.date >= sd) & (ap["date"].dt.date <= ed)].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filt["date"], y=filt["actual"],
            name="Actual", line=dict(color=COOL, width=1.5)))
        fig.add_trace(go.Scatter(x=filt["date"], y=filt["predicted"],
            name="Predicted (OOS)", line=dict(color=WARM, width=1.5)))
        fig.update_layout(**_LAYOUT, yaxis_title="Demand (mcm/d)", height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        filt["residual"] = filt["actual"] - filt["predicted"]
        y_true, y_pred = filt["actual"].values, filt["predicted"].values

    else:
        if bt is None:
            st.caption(
                "Showing in-sample fit. Run full analysis for out-of-sample walk-forward results.")
        else:
            st.caption(
                "In-sample: model predicting data it trained on.  "
                "Useful for spotting systematic patterns, not for quoting accuracy.")

        preds = model.predict(df)
        df = df.copy()
        df["predicted"] = preds

        c1, c2 = st.columns(2)
        sd = c1.date_input("Start", value=df["date"].min().date(),
                           min_value=df["date"].min().date(), max_value=df["date"].max().date())
        ed = c2.date_input("End", value=df["date"].max().date(),
                           min_value=df["date"].min().date(), max_value=df["date"].max().date())

        filt = df[(df["date"].dt.date >= sd) & (df["date"].dt.date <= ed)].copy()
        filt["residual"] = filt["demand_mcm"] - filt["predicted"]
        y_true, y_pred = filt["demand_mcm"].values, filt["predicted"].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filt["date"], y=y_true,
            name="Actual", line=dict(color=COOL, width=1.5)))
        fig.add_trace(go.Scatter(x=filt["date"], y=filt["predicted"],
            name="Predicted (in-sample)", line=dict(color=WARM, width=1.5, dash="dot")))
        fig.update_layout(**_LAYOUT, yaxis_title="Demand (mcm/d)", height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    # Residual bar
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=filt["date"], y=filt["residual"],
        marker_color=np.where(filt["residual"] >= 0, GREEN, RED)))
    fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig2.update_layout(**_LAYOUT, yaxis_title="Residual (mcm/d)", height=220, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    m = compute_metrics(y_true, y_pred)
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{m['rmse_mcm']:.2f} mcm/d")
    c2.metric("MAE", f"{m['mae_mcm']:.1f} mcm/d")
    c3.metric("R²", f"{m['r2']:.4f}")


# == DIAGNOSTICS ========================================================

def page_diagnostics():
    st.header("Model Diagnostics")

    h = _hash()
    model = train_model(h)
    bt = _get_backtest()

    if model is None:
        st.error("Model not available.")
        return

    # Feature importance — always available (fast, from fitted model)
    st.subheader("Feature Importance")
    fi = model.feature_importance()
    fig = px.bar(fi.head(15), x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale=["#2a2015", WARM])
    fig.update_layout(**_LAYOUT, height=400, showlegend=False, coloraxis_showscale=False)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    # Everything below requires the backtest
    if bt is None:
        st.divider()
        _backtest_button("diag")
        return

    # SHAP partial dependence
    st.subheader("Partial Dependence — Top Drivers")
    st.caption(
        "How each feature shifts the prediction while holding others constant.")
    df = _features()
    if df is not None and model.is_fitted:
        try:
            import shap
            explainer = shap.TreeExplainer(model.model)
            X_sample = df[FEATURE_COLS].sample(min(500, len(df)), random_state=42)
            shap_values = explainer.shap_values(X_sample)

            top3 = fi.head(3)["feature"].tolist()
            cols = st.columns(len(top3))
            for idx, feat in enumerate(top3):
                fi_idx = FEATURE_COLS.index(feat)
                with cols[idx]:
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(
                        x=X_sample[feat].values,
                        y=shap_values[:, fi_idx],
                        mode="markers",
                        marker=dict(size=3, color=WARM, opacity=0.4),
                    ))
                    fig_s.update_layout(**_LAYOUT, height=280,
                        xaxis_title=feat, yaxis_title="SHAP value (mcm/d)")
                    st.plotly_chart(fig_s, use_container_width=True)
        except ImportError:
            st.info("Install `shap` for partial dependence plots.")
        except Exception as exc:
            st.warning(f"SHAP computation failed: {exc}")

    # Walk-forward fold table
    st.subheader("Walk-Forward Validation")
    st.dataframe(bt.metrics_table.style.format({
        "rmse_mcm": "{:.2f}", "mape_pct": "{:.1f}", "r2": "{:.4f}", "mae_mcm": "{:.2f}",
    }, na_rep="—"), use_container_width=True)

    ap = bt.all_predictions
    y_true, y_pred = ap["actual"].values, ap["predicted"].values
    resid = y_true - y_pred

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Residuals vs Fitted")
        fig_rf = go.Figure()
        fig_rf.add_trace(go.Scatter(x=y_pred, y=resid, mode="markers",
            marker=dict(size=3, color=WARM, opacity=0.4)))
        fig_rf.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
        fig_rf.update_layout(**_LAYOUT, xaxis_title="Predicted (mcm/d)",
            yaxis_title="Residual (mcm/d)", height=320)
        st.plotly_chart(fig_rf, use_container_width=True)

    with col2:
        st.subheader("Residual Distribution")
        fig_h = px.histogram(x=resid, nbins=50, color_discrete_sequence=[WARM])
        fig_h.update_layout(**_LAYOUT, xaxis_title="Residual (mcm/d)",
            yaxis_title="Count", height=320)
        st.plotly_chart(fig_h, use_container_width=True)

    # ACF of residuals
    st.subheader("Residual Autocorrelation")
    st.caption(
        "Significant bars beyond the confidence band suggest the model "
        "is missing temporal patterns (e.g. multi-day cold spells).")
    acf_vals = residual_acf(resid, nlags=21)
    n = len(resid)
    ci = 1.96 / np.sqrt(n)

    fig_acf = go.Figure()
    for lag, val in enumerate(acf_vals):
        color = WARM if abs(val) > ci or lag == 0 else "rgba(200,202,208,0.4)"
        fig_acf.add_trace(go.Bar(x=[lag], y=[val], marker_color=color,
            showlegend=False, width=0.6))
    fig_acf.add_hline(y=ci, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig_acf.add_hline(y=-ci, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig_acf.add_hline(y=0, line_color="rgba(255,255,255,0.1)")
    fig_acf.update_layout(**_LAYOUT, height=260, xaxis_title="Lag (days)",
        yaxis_title="ACF", bargap=0.15)
    st.plotly_chart(fig_acf, use_container_width=True)

    # Residuals over time
    st.subheader("Residuals Over Time")
    ap2 = ap.copy()
    ap2["residual"] = resid
    fig_t = px.scatter(ap2, x="date", y="residual", color="fold",
        color_continuous_scale="Turbo", opacity=0.4)
    fig_t.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig_t.update_layout(**_LAYOUT, height=300, xaxis_title="Date", yaxis_title="Residual (mcm/d)")
    fig_t.update_traces(marker_size=3)
    st.plotly_chart(fig_t, use_container_width=True)


# == WEATHER IMPACT =====================================================

def page_weather_impact():
    st.header("Weather Impact")

    df = _features()
    if df is None or df.empty:
        st.error("Data not available.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Temperature vs Demand")
        fig = px.scatter(df, x="temp_mean", y="demand_mcm",
            color="is_winter", color_continuous_scale=[COOL, "#e07a5f"],
            opacity=0.4, trendline="ols")
        fig.update_layout(**_LAYOUT, height=380,
            xaxis_title="Mean Temperature (°C)", yaxis_title="Demand (mcm/d)")
        fig.update_traces(marker_size=3)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("HDD vs Demand")
        fig2 = px.scatter(df, x="hdd", y="demand_mcm",
            color="gas_quarter", color_continuous_scale="RdYlBu_r",
            opacity=0.4, trendline="ols")
        fig2.update_layout(**_LAYOUT, height=380,
            xaxis_title="Heating Degree Days", yaxis_title="Demand (mcm/d)")
        fig2.update_traces(marker_size=3)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Wind Speed Effect")
        fig3 = px.scatter(df, x="windspeed_max", y="demand_mcm",
            color="is_winter", color_continuous_scale=[COOL, "#e07a5f"], opacity=0.35)
        fig3.update_layout(**_LAYOUT, height=380,
            xaxis_title="Max Wind Speed (km/h)", yaxis_title="Demand (mcm/d)")
        fig3.update_traces(marker_size=3)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("Wind Chill HDD")
        fig4 = px.scatter(df, x="effective_hdd", y="demand_mcm",
            color="month", color_continuous_scale="Turbo", opacity=0.35)
        fig4.update_layout(**_LAYOUT, height=380,
            xaxis_title="Effective HDD", yaxis_title="Demand (mcm/d)")
        fig4.update_traces(marker_size=3)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Monthly Seasonality")
    monthly = df.groupby("month", as_index=False)["demand_mcm"].agg(["mean", "std"])
    monthly.columns = ["month", "mean", "std"]
    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["name"] = monthly["month"].map(lambda m: names[m - 1])
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=monthly["name"], y=monthly["mean"],
        error_y=dict(type="data", array=monthly["std"], visible=True),
        marker_color=WARM, marker_cornerradius=6))
    fig5.update_layout(**_LAYOUT, height=300, yaxis_title="Demand (mcm/d)")
    st.plotly_chart(fig5, use_container_width=True)


# == FORECAST ===========================================================

def page_forecast():
    st.header("16-Day Demand Forecast")

    h = _hash()
    model = train_model(h)
    if model is None:
        st.error("Model not available.")
        return

    bt = _get_backtest()
    resid = None
    if bt is not None:
        ap = bt.all_predictions
        resid = ap["actual"].values - ap["predicted"].values

    with st.spinner("Generating forecast..."):
        fc = generate_forecast(model, backtest_residuals=resid)

    if fc is None or fc.empty:
        st.warning("Forecast unavailable — weather API may be down.")
        return

    fig = go.Figure()
    if "lower_mcm" in fc.columns:
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper_mcm"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower_mcm"], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(212,132,94,0.1)",
            name="90% interval" if bt else "±10% band"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_mcm"],
        mode="lines+markers", name="Demand forecast",
        line=dict(color=WARM, width=2.5), marker=dict(size=5)))
    fig.update_layout(**_LAYOUT, yaxis_title="Demand (mcm/d)", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    if "temp_mean" in fc.columns:
        st.subheader("Weather Input")
        fig_t = go.Figure()
        if "temp_min" in fc.columns and "temp_max" in fc.columns:
            fig_t.add_trace(go.Scatter(x=fc["date"], y=fc["temp_max"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_t.add_trace(go.Scatter(x=fc["date"], y=fc["temp_min"], mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(91,155,213,0.1)",
                name="Min–Max"))
        fig_t.add_trace(go.Scatter(x=fc["date"], y=fc["temp_mean"],
            mode="lines+markers", name="Mean temp",
            line=dict(color=COOL, width=2), marker=dict(size=4)))
        fig_t.add_hline(y=15.5, line_dash="dot", line_color="rgba(255,255,255,0.2)",
            annotation_text="HDD base", annotation_font_color="rgba(255,255,255,0.3)")
        fig_t.update_layout(**_LAYOUT, yaxis_title="Temperature (°C)", height=260)
        st.plotly_chart(fig_t, use_container_width=True)

    st.subheader("Forecast Table")
    disp = fc.copy()
    disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%Y-%m-%d")
    for c in ["forecast_mcm", "lower_mcm", "upper_mcm", "temp_mean", "temp_min", "temp_max", "hdd"]:
        if c in disp.columns:
            disp[c] = disp[c].round(1)
    st.dataframe(disp, use_container_width=True)

    if bt is not None:
        m = bt.metrics_table[bt.metrics_table["fold"] == "Mean"].iloc[0]
        st.caption(
            f"Intervals from empirical 5th/95th percentile of backtest residuals "
            f"(RMSE={m['rmse_mcm']:.1f} mcm/d, {len(bt.folds)} folds).  "
            f"Not Gaussian — we use the actual error distribution."
        )
        st.caption(
            "Note: backtest RMSE reflects 1-step-ahead accuracy (actual prior-day "
            "demand as lag input). This recursive forecast chains predictions, so "
            "error compounds over the horizon — expect day-7+ accuracy to be "
            "worse than the backtest headline suggests."
        )
    else:
        st.caption(
            "Prediction intervals use a ±10% fallback. Run full analysis for "
            "empirical intervals derived from walk-forward backtest residuals."
        )


# == router =============================================================

{"Overview": page_overview, "Historical Fit": page_historical_fit,
 "Model Diagnostics": page_diagnostics, "Weather Impact": page_weather_impact,
 "Forecast": page_forecast}[page]()
