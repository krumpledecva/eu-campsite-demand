import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EU Campsite Demand Forecaster",
    page_icon="⛺",
    layout="wide",
)

# ── Custom CSS — clean minimal style ─────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #ffffff;
        color: #1a1a1a;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding: 2.5rem 4rem 2rem 4rem;
        max-width: 100%;
    }

    /* Hero title */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #111111;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1rem;
        color: #666666;
        margin-bottom: 0rem;
    }

    /* Thin divider */
    .divider {
        border: none;
        border-top: 1px solid #e8e8e8;
        margin: 1.5rem 0;
    }

    /* Section label */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #999999;
        margin-bottom: 0.6rem;
    }

    /* Lag info box */
    .lag-box {
        background: #f9f9f9;
        border: 1px solid #eeeeee;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.9rem;
        color: #444;
    }
    .lag-ok { color: #2e7d32; font-weight: 600; }
    .lag-missing { color: #b71c1c; }

    /* Result cards */
    .result-card {
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 1.6rem 1.8rem;
        text-align: center;
        background: #ffffff;
    }
    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #999999;
        margin-bottom: 0.8rem;
    }
    .result-value-large {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.1;
        color: #111111;
    }
    .badge-low    { color: #2e7d32; }
    .badge-medium { color: #e65100; }
    .badge-high   { color: #b71c1c; }

    /* Probability bar row */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.45rem;
        font-size: 0.88rem;
    }
    .prob-label { width: 52px; color: #555; }
    .prob-bar-bg {
        flex: 1;
        background: #f0f0f0;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    .prob-pct { width: 36px; text-align: right; color: #333; font-weight: 600; }

    /* Footer */
    .app-footer {
        font-size: 0.75rem;
        color: #aaaaaa;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eeeeee;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models and data ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return joblib.load("classifier.pkl"), joblib.load("regressor.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("campsite_data.csv")

clf_model, reg_model = load_models()
hist = load_data()

COUNTRIES   = sorted(hist["geo"].unique().tolist())
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
PROB_COLORS = ["#2e7d32", "#e65100", "#b71c1c"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">⛺ EU Campsite Demand Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Predict demand level and overnight stays for any European country and month.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Select parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("FR"), label_visibility="collapsed")
with col2:
    year = st.number_input("Year", min_value=2010, max_value=2030, value=2025, label_visibility="collapsed")
with col3:
    month = st.selectbox("Month", range(1, 13),
                         format_func=lambda m: MONTH_NAMES[m - 1],
                         index=6, label_visibility="collapsed")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Lag lookup ────────────────────────────────────────────────────────────────
def lookup_lag(geo, yr, mo, shift):
    mo2, yr2 = mo - shift, yr
    while mo2 < 1:
        mo2 += 12; yr2 -= 1
    row = hist[(hist["geo"] == geo) & (hist["year"] == yr2) & (hist["month"] == mo2)]
    return float(row["nights_spent"].values[0]) if len(row) else None

lag_1m_auto  = lookup_lag(country, year, month, 1)
lag_12m_auto = lookup_lag(country, year, month, 12)

st.markdown('<div class="section-label">Lag features</div>', unsafe_allow_html=True)

lcol1, lcol2 = st.columns(2)
with lcol1:
    if lag_1m_auto is not None:
        st.markdown(f'<div class="lag-box"><span class="lag-ok">✓ Last month auto-loaded</span><br>{lag_1m_auto:,.0f} nights</div>', unsafe_allow_html=True)
        lag_1m = lag_1m_auto
    else:
        st.markdown('<div class="section-label">Last month nights spent (enter manually)</div>', unsafe_allow_html=True)
        lag_1m = st.number_input("lag_1m", min_value=0, value=50000, step=1000, label_visibility="collapsed")

with lcol2:
    if lag_12m_auto is not None:
        st.markdown(f'<div class="lag-box"><span class="lag-ok">✓ Same month last year auto-loaded</span><br>{lag_12m_auto:,.0f} nights</div>', unsafe_allow_html=True)
        lag_12m = lag_12m_auto
    else:
        st.markdown('<div class="section-label">Same month last year (enter manually)</div>', unsafe_allow_html=True)
        lag_12m = st.number_input("lag_12m", min_value=0, value=50000, step=1000, label_visibility="collapsed")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
predict = st.button("Predict demand →", type="primary", use_container_width=True)

if predict:
    X = pd.DataFrame([{
        "geo":       country,
        "year_rel":  year - 2010,
        "month":     month,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "lag_1m":    lag_1m,
        "lag_12m":   lag_12m,
        "is_summer": int(month in [6, 7, 8]),
        "is_july":   int(month == 7),
        "is_covid":  int(year in [2020, 2021]),
    }])

    demand_code  = int(clf_model.predict(X)[0])
    demand_proba = clf_model.predict_proba(X)[0]
    nights_pred  = max(0, float(reg_model.predict(X)[0]))

    labels       = ["Low", "Medium", "High"]
    badge_class  = ["badge-low", "badge-medium", "badge-high"]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns([2, 0.15, 2])

    with rc1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Demand Level</div>
            <div class="result-value-large {badge_class[demand_code]}">
                {labels[demand_code]}
            </div>
            <div style="margin-top:1.4rem;">
        """, unsafe_allow_html=True)

        for i, (lbl, prob) in enumerate(zip(labels, demand_proba)):
            pct = int(prob * 100)
            st.markdown(f"""
            <div class="prob-row">
                <span class="prob-label">{lbl}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct}%;background:{PROB_COLORS[i]};"></div>
                </div>
                <span class="prob-pct">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    with rc3:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Nights Spent</div>
            <div class="result-value-large" style="color:#111111;">
                {nights_pred:,.0f}
            </div>
            <div style="margin-top:0.6rem;font-size:0.82rem;color:#999;">
                {country} &middot; {MONTH_NAMES[month-1]} {year}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Historical chart ──────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(f'<div class="section-label">Historical trend — {country}</div>', unsafe_allow_html=True)

country_hist = hist[hist["geo"] == country].copy()
country_hist["date"] = pd.to_datetime(country_hist[["year","month"]].assign(day=1))
country_hist = country_hist.sort_values("date")

if len(country_hist) > 0:
    fig, ax = plt.subplots(figsize=(14, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(country_hist["date"], country_hist["nights_spent"] / 1e6,
            color="#1a1a1a", linewidth=1.2, zorder=3)
    ax.fill_between(country_hist["date"], country_hist["nights_spent"] / 1e6,
                    alpha=0.07, color="#1a1a1a")

    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
               alpha=0.08, color="#b71c1c", zorder=1)
    ax.text(pd.Timestamp("2020-06-01"), ax.get_ylim()[1] * 0.92,
            "COVID-19", fontsize=7.5, color="#b71c1c", va="top")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.set_xlabel("")
    ax.set_ylabel("Nights spent", fontsize=8.5, color="#888")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0, color="#eeeeee", linewidth=0.8)
    ax.grid(axis="y", color="#f0f0f0", linewidth=0.8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">Data: Eurostat tour_occ_nim (CC BY 4.0) &nbsp;·&nbsp; '
    'Model: XGBoost trained on 7,104 country-month observations &nbsp;·&nbsp; '
    'Author: Nuša Brezovnik Bunderla</div>',
    unsafe_allow_html=True
)
