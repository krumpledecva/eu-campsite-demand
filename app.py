import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import folium
from streamlit_folium import st_folium
import requests

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
html, body, [class*="css"], p, div, span, label, input, select, textarea {
    font-family: 'Nunito', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Page padding */
.block-container {
    padding: 1.5rem 3.5rem 2rem 3.5rem;
    max-width: 100%;
}

/* ── Animations ── */
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.12); }
}
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-6px); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1B4332 0%, #0D1F17 100%);
    border-radius: 16px;
    padding: 2.2rem 2.8rem 2rem 2.8rem;
    margin-bottom: 0.5rem;
    border: 1px solid #2d6a4f;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-emoji {
    font-size: 3rem;
    display: inline-block;
    animation: bounce 2.5s ease-in-out infinite;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #F0EAD6;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
    line-height: 1.15;
}
.hero-sub {
    font-size: 1rem;
    color: #95D5B2;
    font-weight: 400;
}
.golden-divider {
    border: none;
    border-top: 2px solid #F4A261;
    margin: 1.4rem 0;
    opacity: 0.6;
}

/* ── Section headers ── */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #F0EAD6;
    border-bottom: 2px solid #1B4332;
    padding-bottom: 0.35rem;
    margin-bottom: 1rem;
}

/* ── Lag box ── */
.lag-box {
    background: #1B4332;
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    font-size: 0.9rem;
    color: #F0EAD6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.lag-ok      { color: #95D5B2; font-weight: 700; }
.lag-missing { color: #F4A261; }

/* ── Result cards ── */
.result-card {
    background: #1B4332;
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    animation: fadeIn 0.5s ease-out;
}
.result-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #95D5B2;
    margin-bottom: 0.8rem;
}
.result-value-large {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
}
.badge-low    { color: #95D5B2; }
.badge-medium { color: #52B788; }
.badge-high   { color: #F4A261; }

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}
.prob-label { width: 56px; color: #95D5B2; font-weight: 600; }
.prob-bar-bg {
    flex: 1;
    background: #0D1F17;
    border-radius: 6px;
    height: 9px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 6px; }
.prob-pct { width: 36px; text-align: right; color: #F0EAD6; font-weight: 700; }

/* ── Chart wrapper ── */
.chart-card {
    background: #1B4332;
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    margin-bottom: 0.5rem;
}

/* ── Streamlit button ── */
div.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700;
    font-size: 1.05rem;
    border-radius: 12px;
    padding: 0.65rem 1.5rem;
    background: #F4A261;
    color: #0D1F17;
    border: none;
    box-shadow: 0 4px 14px rgba(244,162,97,0.35);
    transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
}
div.stButton > button:hover {
    background: #e8894a;
    box-shadow: 0 6px 20px rgba(244,162,97,0.5);
    transform: translateY(-1px);
}

/* ── Campsite cards ── */
.camp-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1rem;
    margin-top: 0.8rem;
}
.camp-card {
    background: #1B4332;
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    animation: fadeIn 0.4s ease-out;
}
.camp-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #F0EAD6;
    margin-bottom: 0.3rem;
}
.camp-stars { color: #F4A261; font-size: 0.9rem; margin-bottom: 0.3rem; }
.camp-region { font-size: 0.78rem; color: #95D5B2; margin-bottom: 0.6rem; }
.camp-link {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 700;
    color: #0D1F17;
    background: #F4A261;
    border-radius: 8px;
    padding: 0.3rem 0.75rem;
    text-decoration: none;
}
.camp-link:hover { background: #e8894a; }
.camp-none { color: #95D5B2; font-size: 0.9rem; padding: 0.5rem 0; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    font-size: 0.78rem;
    color: #F0EAD6;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 2px solid #F4A261;
    opacity: 0.7;
}

/* ── Mobile ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem 1.5rem 1rem; }
    .hero-title { font-size: 1.5rem; }
    .hero-banner { padding: 1.4rem 1.4rem 1.2rem 1.4rem; }
    .result-card { padding: 1.2rem 1rem; }
    .result-value-large { font-size: 2rem; }
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

# ── Campsite lookup via OpenStreetMap Overpass API ───────────────────────────
EUROSTAT_TO_ISO = {"EL": "GR", "UK": "GB"}

@st.cache_data(ttl=3600, show_spinner=False)
def get_campsites(country_code, limit=9):
    iso = EUROSTAT_TO_ISO.get(country_code, country_code)
    query = f"""
    [out:json][timeout:25];
    area["ISO3166-1"="{iso}"]->.c;
    node["tourism"="camp_site"]["name"](area.c);
    out body {limit};
    """
    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=25,
        )
        items = []
        for el in r.json().get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name", "").strip()
            if not name:
                continue
            raw_stars = tags.get("stars", tags.get("rating", ""))
            try:
                n = int(float(raw_stars))
                stars_str = "⭐" * min(n, 5)
            except (ValueError, TypeError):
                stars_str = "—"
            region = (tags.get("addr:city") or
                      tags.get("addr:county") or
                      tags.get("addr:region") or "")
            booking = (
                "https://www.booking.com/searchresults.html"
                f"?ss={requests.utils.quote(name + ' ' + country_code)}&label=campsite"
            )
            items.append({
                "name":    name,
                "lat":     el.get("lat"),
                "lon":     el.get("lon"),
                "stars":   stars_str,
                "region":  region,
                "booking": booking,
            })
        return items
    except Exception:
        return []

COUNTRIES   = sorted(hist["geo"].unique().tolist())
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
PROB_COLORS = ["#95D5B2", "#52B788", "#F4A261"]

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-emoji">⛺</div>
  <div class="hero-title">EU Campsite Demand Forecaster</div>
  <div class="hero-sub">Predict demand level and overnight stays for any European country and month.</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr class="golden-divider">', unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⛺ Select Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("FR"), label_visibility="visible")
with col2:
    year = st.number_input("Year", min_value=2010, max_value=2030, value=2025, label_visibility="visible")
with col3:
    month = st.selectbox("Month", range(1, 13),
                         format_func=lambda m: MONTH_NAMES[m - 1],
                         index=6, label_visibility="visible")

st.markdown('<hr class="golden-divider">', unsafe_allow_html=True)

# ── Lag lookup ────────────────────────────────────────────────────────────────
def lookup_lag(geo, yr, mo, shift):
    mo2, yr2 = mo - shift, yr
    while mo2 < 1:
        mo2 += 12; yr2 -= 1
    row = hist[(hist["geo"] == geo) & (hist["year"] == yr2) & (hist["month"] == mo2)]
    return float(row["nights_spent"].values[0]) if len(row) else None

lag_1m_auto  = lookup_lag(country, year, month, 1)
lag_12m_auto = lookup_lag(country, year, month, 12)

st.markdown('<div class="section-header">📊 Lag Features</div>', unsafe_allow_html=True)

lcol1, lcol2 = st.columns(2)
with lcol1:
    if lag_1m_auto is not None:
        st.markdown(f'<div class="lag-box"><span class="lag-ok">✓ Last month auto-loaded</span><br>{lag_1m_auto:,.0f} nights</div>', unsafe_allow_html=True)
        lag_1m = lag_1m_auto
    else:
        lag_1m = st.number_input("Last month nights spent", min_value=0, value=50000, step=1000)

with lcol2:
    if lag_12m_auto is not None:
        st.markdown(f'<div class="lag-box"><span class="lag-ok">✓ Same month last year auto-loaded</span><br>{lag_12m_auto:,.0f} nights</div>', unsafe_allow_html=True)
        lag_12m = lag_12m_auto
    else:
        lag_12m = st.number_input("Same month last year", min_value=0, value=50000, step=1000)

st.markdown('<hr class="golden-divider">', unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
_, bcol, _ = st.columns([1, 2, 1])
with bcol:
    predict = st.button("Predict 🔍", type="primary", use_container_width=True)

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

    country_avg = hist[(hist["geo"] == country) & (hist["month"] == month)]["nights_spent"].mean()
    if nights_pred > country_avg * 1.1:
        interp = f"Above average for {country} in {MONTH_NAMES[month-1]}"
    elif nights_pred < country_avg * 0.9:
        interp = f"Below average for {country} in {MONTH_NAMES[month-1]}"
    else:
        interp = f"Around average for {country} in {MONTH_NAMES[month-1]}"

    st.markdown('<hr class="golden-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔍 Results</div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns([2, 0.15, 2])

    with rc1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Demand Level</div>
            <div class="result-value-large {badge_class[demand_code]}">{labels[demand_code]}</div>
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
            <div class="result-value-large" style="color:#F4A261;">
                ⛺ {nights_pred:,.0f}
            </div>
            <div style="margin-top:0.8rem;font-size:0.88rem;color:#95D5B2;font-weight:600;">
                {interp}
            </div>
            <div style="margin-top:0.4rem;font-size:0.78rem;color:#52B788;">
                {country} · {MONTH_NAMES[month-1]} {year}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Historical chart ──────────────────────────────────────────────────────────
st.markdown('<hr class="golden-divider">', unsafe_allow_html=True)
st.markdown(f'<div class="section-header">📊 Historical Trend — {country}</div>', unsafe_allow_html=True)

country_hist = hist[hist["geo"] == country].copy()
country_hist["date"] = pd.to_datetime(country_hist[["year","month"]].assign(day=1))
country_hist = country_hist.sort_values("date")

if len(country_hist) > 0:
    fig, ax = plt.subplots(figsize=(14, 3.4))
    fig.patch.set_facecolor("#1B4332")
    ax.set_facecolor("#1B4332")

    ax.plot(country_hist["date"], country_hist["nights_spent"] / 1e6,
            color="#F4A261", linewidth=1.8, zorder=3)
    ax.fill_between(country_hist["date"], country_hist["nights_spent"] / 1e6,
                    alpha=0.15, color="#F4A261")

    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
               alpha=0.15, color="#95D5B2", zorder=1)
    ax.text(pd.Timestamp("2020-06-01"), ax.get_ylim()[1] * 0.92,
            "COVID-19", fontsize=7.5, color="#95D5B2", va="top")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.set_xlabel("")
    ax.set_ylabel("Nights spent", fontsize=8.5, color="#95D5B2")
    ax.tick_params(colors="#95D5B2", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#2d6a4f", linewidth=0.8)

    plt.tight_layout()
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
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

st.markdown('<hr class="golden-divider">', unsafe_allow_html=True)
st.markdown(f'<div class="section-header">🗺️ Campsites in {country}</div>', unsafe_allow_html=True)

with st.spinner("Finding campsites via OpenStreetMap..."):
    campsites = get_campsites(country)

lat, lon, zoom = COUNTRY_COORDS.get(country, (54.0, 15.0, 4))
m = folium.Map(
    location=[lat, lon],
    zoom_start=zoom,
    tiles="CartoDB dark_matter",
    zoom_control=True,
    scrollWheelZoom=False,
)

# Country centre marker
folium.Marker(
    location=[lat, lon],
    tooltip=country,
    icon=folium.Icon(color="orange", icon="star"),
).add_to(m)

# Campsite pins
for cs in campsites:
    if cs["lat"] and cs["lon"]:
        folium.Marker(
            location=[cs["lat"], cs["lon"]],
            tooltip=cs["name"],
            popup=folium.Popup(
                f"<b>{cs['name']}</b><br>{cs['stars']}<br>"
                f"<a href='{cs['booking']}' target='_blank'>View prices →</a>",
                max_width=200,
            ),
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(m)

st_folium(m, use_container_width=True, height=420, returned_objects=[])

# ── Campsite list ─────────────────────────────────────────────────────────────
st.markdown(f'<div class="section-header">⛺ Top Campsites — {country}</div>', unsafe_allow_html=True)

if campsites:
    cards_html = '<div class="camp-grid">'
    for cs in campsites:
        region_line = f'<div class="camp-region">📍 {cs["region"]}</div>' if cs["region"] else ""
        stars_line  = f'<div class="camp-stars">{cs["stars"]}</div>' if cs["stars"] != "—" else ""
        cards_html += f"""
        <div class="camp-card">
            <div class="camp-name">{cs['name']}</div>
            {stars_line}
            {region_line}
            <a class="camp-link" href="{cs['booking']}" target="_blank">View prices →</a>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
else:
    st.markdown('<div class="camp-none">No campsites found for this country in OpenStreetMap.</div>',
                unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">Built with ⛺ by Nuša Brezovnik Bunderla &nbsp;·&nbsp; Data: Eurostat CC BY 4.0</div>',
    unsafe_allow_html=True
)
