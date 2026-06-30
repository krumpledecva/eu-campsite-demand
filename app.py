import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EU Campsite Demand Forecaster",
    page_icon="⛺",
    layout="centered",
)

# ── Load models and historical data ──────────────────────────────────────────
@st.cache_resource
def load_models():
    clf = joblib.load("classifier.pkl")
    reg = joblib.load("regressor.pkl")
    return clf, reg

@st.cache_data
def load_data():
    df = pd.read_csv("campsite_data.csv")
    return df

clf_model, reg_model = load_models()
hist = load_data()

COUNTRIES = sorted(hist["geo"].unique().tolist())
DEMAND_LABELS = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
DEMAND_COLORS = {0: "#c6efce", 1: "#ffeb9c", 2: "#ffc7ce"}
MONTH_NAMES   = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("⛺ EU Campsite Demand Forecaster")
st.markdown(
    "Predict **demand level** (Low / Medium / High) and "
    "**estimated nights spent** at EU camping grounds for any country and month. "
    "Powered by XGBoost trained on Eurostat `tour_occ_nim` data."
)
st.divider()

# ── Inputs ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("FR"))

with col2:
    year = st.number_input("Year", min_value=2010, max_value=2030, value=2025)

with col3:
    month = st.selectbox("Month", range(1, 13),
                          format_func=lambda m: MONTH_NAMES[m - 1], index=6)

# ── Lag feature lookup ───────────────────────────────────────────────────────
def lookup_lag(geo, yr, mo, shift_months):
    """Return nights_spent for geo at (yr, mo) shifted back by shift_months."""
    target_mo = mo - shift_months
    target_yr = yr
    while target_mo < 1:
        target_mo += 12
        target_yr -= 1
    row = hist[(hist["geo"] == geo) &
               (hist["year"] == target_yr) &
               (hist["month"] == target_mo)]
    return float(row["nights_spent"].values[0]) if len(row) else None

lag_1m_auto  = lookup_lag(country, year, month, 1)
lag_12m_auto = lookup_lag(country, year, month, 12)

st.markdown("#### Lag features")
lag_col1, lag_col2 = st.columns(2)

with lag_col1:
    if lag_1m_auto is not None:
        st.success(f"Last month auto-loaded: **{lag_1m_auto:,.0f} nights**")
        lag_1m = lag_1m_auto
    else:
        lag_1m = st.number_input(
            "Last month nights spent (enter manually)",
            min_value=0, value=50000, step=1000,
        )

with lag_col2:
    if lag_12m_auto is not None:
        st.success(f"Same month last year auto-loaded: **{lag_12m_auto:,.0f} nights**")
        lag_12m = lag_12m_auto
    else:
        lag_12m = st.number_input(
            "Same month last year (enter manually)",
            min_value=0, value=50000, step=1000,
        )

# ── Build feature row ────────────────────────────────────────────────────────
def build_features(geo, yr, mo, l1m, l12m):
    return pd.DataFrame([{
        "geo":       geo,
        "year_rel":  yr - 2010,
        "month":     mo,
        "month_sin": np.sin(2 * np.pi * mo / 12),
        "month_cos": np.cos(2 * np.pi * mo / 12),
        "lag_1m":    l1m,
        "lag_12m":   l12m,
        "is_summer": int(mo in [6, 7, 8]),
        "is_july":   int(mo == 7),
        "is_covid":  int(yr in [2020, 2021]),
    }])

# ── Predict ──────────────────────────────────────────────────────────────────
st.divider()

if st.button("Predict demand", type="primary", use_container_width=True):
    X = build_features(country, year, month, lag_1m, lag_12m)

    demand_code  = int(clf_model.predict(X)[0])
    demand_proba = clf_model.predict_proba(X)[0]
    nights_pred  = max(0, float(reg_model.predict(X)[0]))

    # Results
    st.markdown("### Results")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        color = DEMAND_COLORS[demand_code]
        label = DEMAND_LABELS[demand_code]
        st.markdown(
            f"<div style='background:{color};padding:20px;border-radius:8px;"
            f"text-align:center;font-size:1.4em;font-weight:bold'>"
            f"Demand level<br>{label}</div>",
            unsafe_allow_html=True,
        )

    with res_col2:
        st.metric(
            label="Estimated nights spent",
            value=f"{nights_pred:,.0f}",
            help="XGBoost regression prediction",
        )

    # Probability breakdown
    st.markdown("**Demand level probabilities**")
    prob_df = pd.DataFrame({
        "Level": ["Low", "Medium", "High"],
        "Probability": demand_proba,
    })
    st.bar_chart(prob_df.set_index("Level"))

# ── Historical trend chart ────────────────────────────────────────────────────
st.divider()
st.markdown(f"### Historical trend — {country}")

country_hist = hist[hist["geo"] == country].copy()
country_hist["date"] = pd.to_datetime(
    country_hist[["year", "month"]].assign(day=1)
)
country_hist = country_hist.sort_values("date")

if len(country_hist) > 0:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(country_hist["date"], country_hist["nights_spent"] / 1e6,
            color="steelblue", linewidth=1)
    ax.fill_between(country_hist["date"],
                    country_hist["nights_spent"] / 1e6,
                    alpha=0.15, color="steelblue")
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
               alpha=0.12, color="red", label="COVID period")
    ax.set_ylabel("Nights spent (millions)")
    ax.set_xlabel("")
    ax.set_title(f"Monthly campsite nights — {country} (all years)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("No historical data available for this country.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data: Eurostat `tour_occ_nim` (CC BY 4.0) · "
    "Model: XGBoost trained on 7,104 country-month observations · "
    "Author: Nuša Brezovnik Bunderla"
)
