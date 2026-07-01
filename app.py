import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import folium
from streamlit_folium import st_folium

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EU Campsite Demand Forecaster",
    page_icon="⛺",
    layout="wide",
)

# ── All custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');

/* Global */
html, body, [class*="css"], p, div, span, label, input, select {
    font-family: 'Nunito', sans-serif !important;
}
body { background-color: #f7f4f0; color: #2d2d2d; }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Page padding */
.block-container {
    padding: 1.5rem 3.5rem 2rem 3.5rem;
    max-width: 100%;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #e2ddd8;
    margin: 1.4rem 0;
}

/* Section label */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #a89f96;
    margin-bottom: 0.6rem;
}

/* Lag box */
.lag-box {
    background: #ffffff;
    border: 1px solid #e8e2db;
    border-radius: 12px;
    padding: 0.8rem 1.1rem;
    font-size: 0.9rem;
    color: #555;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.lag-ok      { color: #3a7d44; font-weight: 700; }
.lag-missing { color: #c0392b; }

/* Result cards */
.result-card {
    border: 1px solid #e8e2db;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
    background: #ffffff;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
}
.result-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #a89f96;
    margin-bottom: 0.8rem;
}
.result-value-large {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
}
.badge-low    { color: #3a7d44; }
.badge-medium { color: #d4720a; }
.badge-high   { color: #c0392b; }

/* Probability bars */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}
.prob-label { width: 52px; color: #777; font-weight: 600; }
.prob-bar-bg {
    flex: 1;
    background: #f0ece8;
    border-radius: 6px;
    height: 9px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 6px; }
.prob-pct { width: 36px; text-align: right; color: #444; font-weight: 700; }

/* Streamlit button */
div.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700;
    font-size: 1rem;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    background: #3a7d44;
    color: #ffffff;
    border: none;
    box-shadow: 0 4px 12px rgba(58,125,68,0.25);
    transition: background 0.2s, box-shadow 0.2s;
}
div.stButton > button:hover {
    background: #2f6638;
    box-shadow: 0 6px 18px rgba(58,125,68,0.35);
}

/* Footer */
.app-footer {
    font-size: 0.75rem;
    color: #b0a89f;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e2ddd8;
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
PROB_COLORS = ["#3a7d44", "#d4720a", "#c0392b"]

# ── Hero banner (components.html renders raw SVG) ─────────────────────────────
components.html("""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700;800&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#f7f4f0; font-family:'Nunito',sans-serif; }
.banner { background:#fff5f7; border-radius:16px; overflow:hidden;
          box-shadow:0 4px 20px rgba(0,0,0,0.07); }
.banner svg { width:100%; display:block; }
.hero-content { padding:1.2rem 2.5rem 1.8rem 2.5rem; background:#fff5f7; }
.hero-title { font-size:2rem; font-weight:800; color:#2d2d2d;
              margin-bottom:0.3rem; letter-spacing:-0.5px; }
.hero-sub   { font-size:1rem; color:#7a6f69; font-weight:400; }
</style>
</head>
<body>
<div class="banner">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 180">
    <rect width="1200" height="180" fill="#fff5f7"/>
    <polygon points="0,160 150,45 300,160"    fill="none" stroke="#e0c8cf" stroke-width="1.5"/>
    <polygon points="200,160 390,18 570,160"  fill="none" stroke="#e0c8cf" stroke-width="1.5"/>
    <polygon points="500,160 690,50 870,160"  fill="none" stroke="#e0c8cf" stroke-width="1.5"/>
    <polygon points="760,160 960,28 1110,160" fill="none" stroke="#e0c8cf" stroke-width="1.5"/>
    <polygon points="980,160 1130,62 1200,160" fill="none" stroke="#e0c8cf" stroke-width="1.5"/>
    <rect x="0" y="155" width="1200" height="25" fill="#f9eef1"/>
    <circle cx="80" cy="42" r="22" fill="#ffffff" stroke="#2d2d2d" stroke-width="1.8"/>
    <circle cx="91" cy="36" r="18" fill="#fff5f7"/>
    <circle cx="28"  cy="18" r="2"   fill="#2d2d2d"/>
    <circle cx="155" cy="14" r="1.5" fill="#2d2d2d"/>
    <circle cx="238" cy="28" r="2"   fill="#2d2d2d"/>
    <circle cx="950" cy="14" r="2"   fill="#2d2d2d"/>
    <circle cx="1055" cy="38" r="1.5" fill="#2d2d2d"/>
    <circle cx="1100" cy="18" r="2"  fill="#2d2d2d"/>
    <circle cx="1155" cy="32" r="1.5" fill="#FFB3C6"/>
    <rect x="178" y="138" width="9" height="22" fill="#2d2d2d"/>
    <polygon points="182,78 200,140 164,140"  fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <polygon points="182,58 206,108 158,108"  fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <rect x="220" y="142" width="8" height="18" fill="#2d2d2d"/>
    <polygon points="224,90 240,142 208,142"  fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <polygon points="224,72 244,114 204,114"  fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <polygon points="420,72 515,155 325,155"  fill="#FFD6E0" stroke="#2d2d2d" stroke-width="2.5"/>
    <line x1="420" y1="72" x2="420" y2="155" stroke="#2d2d2d" stroke-width="1.5" stroke-dasharray="5,3"/>
    <polygon points="420,118 447,155 393,155" fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <line x1="325" y1="155" x2="310" y2="170" stroke="#2d2d2d" stroke-width="1.5"/>
    <line x1="515" y1="155" x2="530" y2="170" stroke="#2d2d2d" stroke-width="1.5"/>
    <line x1="570" y1="154" x2="582" y2="154" stroke="#2d2d2d" stroke-width="2"/>
    <line x1="576" y1="143" x2="570" y2="154" stroke="#2d2d2d" stroke-width="1.5"/>
    <line x1="576" y1="143" x2="582" y2="154" stroke="#2d2d2d" stroke-width="1.5"/>
    <ellipse cx="576" cy="141" rx="4" ry="7"  fill="#FFD6E0" stroke="#2d2d2d" stroke-width="1.5"/>
    <rect x="610" y="140" width="9" height="20" fill="#2d2d2d"/>
    <polygon points="614,80 634,142 594,142"  fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <polygon points="614,60 638,106 590,106"  fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <rect x="680" y="108" width="215" height="52" rx="5" fill="#FFD6E0" stroke="#2d2d2d" stroke-width="2.5"/>
    <rect x="848" y="88"  width="50"  height="25" rx="4" fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <rect x="856" y="93"  width="32"  height="16" rx="3" fill="#e8f4ff" stroke="#2d2d2d" stroke-width="1.5"/>
    <rect x="696" y="116" width="36"  height="22" rx="3" fill="#ffffff" stroke="#2d2d2d" stroke-width="1.5"/>
    <rect x="742" y="116" width="26"  height="22" rx="3" fill="#ffffff" stroke="#2d2d2d" stroke-width="1.5"/>
    <rect x="778" y="110" width="30"  height="42" rx="3" fill="#ffffff" stroke="#2d2d2d" stroke-width="1.5"/>
    <circle cx="805" cy="132" r="2.5" fill="#2d2d2d"/>
    <rect x="684" y="105" width="162" height="5"  rx="2" fill="#2d2d2d"/>
    <line x1="684" y1="110" x2="638" y2="133" stroke="#2d2d2d" stroke-width="1.5"/>
    <rect x="636" y="108" width="50"  height="6"  rx="2" fill="#FFD6E0" stroke="#2d2d2d" stroke-width="1.5"/>
    <line x1="638" y1="114" x2="638" y2="133" stroke="#2d2d2d" stroke-width="1.5"/>
    <circle cx="722" cy="163" r="14" fill="#2d2d2d"/>
    <circle cx="722" cy="163" r="7"  fill="#888888"/>
    <circle cx="722" cy="163" r="3"  fill="#ffffff"/>
    <circle cx="848" cy="163" r="14" fill="#2d2d2d"/>
    <circle cx="848" cy="163" r="7"  fill="#888888"/>
    <circle cx="848" cy="163" r="3"  fill="#ffffff"/>
    <rect x="950" y="138" width="9"  height="22" fill="#2d2d2d"/>
    <polygon points="954,78 974,140 934,140"   fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <polygon points="954,58 978,106 930,106"   fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <rect x="992" y="144" width="8"  height="16" fill="#2d2d2d"/>
    <polygon points="996,94 1012,146 980,146"  fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <polygon points="996,76 1016,118 976,118"  fill="#ffffff" stroke="#2d2d2d" stroke-width="2"/>
    <path d="M 0,168 Q 600,158 1200,168" stroke="#e8c8d0" stroke-width="1" fill="none" stroke-dasharray="18,10"/>
  </svg>
  <div class="hero-content">
    <div class="hero-title">&#x26FA; EU Campsite Demand Forecaster</div>
    <div class="hero-sub">Predict demand level and overnight stays for any European country and month.</div>
  </div>
</div>
</body>
</html>
""", height=340, scrolling=False)

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

    labels      = ["Low", "Medium", "High"]
    badge_class = ["badge-low", "badge-medium", "badge-high"]

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
            <div class="result-value-large" style="color:#2d2d2d;">
                {nights_pred:,.0f}
            </div>
            <div style="margin-top:0.6rem;font-size:0.82rem;color:#a89f96;">
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
    fig.patch.set_facecolor("#f7f4f0")
    ax.set_facecolor("#f7f4f0")

    ax.plot(country_hist["date"], country_hist["nights_spent"] / 1e6,
            color="#3a7d44", linewidth=1.5, zorder=3)
    ax.fill_between(country_hist["date"], country_hist["nights_spent"] / 1e6,
                    alpha=0.1, color="#3a7d44")

    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
               alpha=0.08, color="#c0392b", zorder=1)
    ax.text(pd.Timestamp("2020-06-01"), ax.get_ylim()[1] * 0.92,
            "COVID-19", fontsize=7.5, color="#c0392b", va="top")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.set_xlabel("")
    ax.set_ylabel("Nights spent", fontsize=8.5, color="#a89f96")
    ax.tick_params(colors="#c0b8b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#e8e2db", linewidth=0.8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Map ───────────────────────────────────────────────────────────────────────
COUNTRY_COORDS = {
    "AT": (47.5, 14.5, 7), "BE": (50.5,  4.5, 8), "BG": (42.7, 25.5, 7),
    "CH": (46.8,  8.2, 8), "CY": (35.1, 33.4, 9), "CZ": (49.8, 15.5, 7),
    "DE": (51.2, 10.4, 6), "DK": (56.3,  9.5, 7), "EE": (58.6, 25.0, 7),
    "EL": (39.1, 21.8, 7), "ES": (40.5, -3.7, 6), "FI": (64.0, 26.0, 5),
    "FR": (46.2,  2.2, 6), "HR": (45.1, 15.2, 7), "HU": (47.2, 19.5, 7),
    "IE": (53.4, -8.2, 7), "IS": (64.9,-19.0, 6), "IT": (41.9, 12.6, 6),
    "LT": (55.2, 23.9, 7), "LU": (49.8,  6.1,10), "LV": (56.9, 24.6, 7),
    "ME": (42.7, 19.4, 8), "MK": (41.6, 21.7, 8), "MT": (35.9, 14.5,11),
    "NL": (52.4,  4.9, 8), "NO": (64.5, 17.5, 5), "PL": (51.9, 19.1, 6),
    "PT": (39.4, -8.2, 7), "RO": (45.9, 24.9, 7), "RS": (44.0, 21.0, 7),
    "SE": (60.1, 18.6, 5), "SI": (46.1, 14.8, 8), "SK": (48.7, 19.7, 8),
    "TR": (38.9, 35.2, 6), "UK": (55.4, -3.4, 6),
}

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(f'<div class="section-label">Location — {country}</div>', unsafe_allow_html=True)

lat, lon, zoom = COUNTRY_COORDS.get(country, (54.0, 15.0, 4))
m = folium.Map(
    location=[lat, lon],
    zoom_start=zoom,
    tiles="CartoDB positron",
    zoom_control=True,
    scrollWheelZoom=False,
)
folium.Marker(
    location=[lat, lon],
    tooltip=country,
    icon=folium.Icon(color="green", icon="star"),
).add_to(m)
st_folium(m, use_container_width=True, height=380, returned_objects=[])

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">Data: Eurostat tour_occ_nim (CC BY 4.0) &nbsp;·&nbsp; '
    'Model: XGBoost trained on 7,104 country-month observations &nbsp;·&nbsp; '
    'Author: Nuša Brezovnik Bunderla</div>',
    unsafe_allow_html=True
)
