import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import folium
from pathlib import Path
from streamlit_folium import st_folium

# ── Helper: base64-encode local image for CSS background ─────────────────────
def img_b64(path):
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EU Campsite Demand Forecaster",
    page_icon="⛺",
    layout="wide",
)

# ── Hero background: photo if available, gradient fallback ───────────────────
hero_b64  = img_b64("assets/hero.jpg")
hero_bg   = (f"url('data:image/jpeg;base64,{hero_b64}')"
             if hero_b64 else "none")
hero_grad = "linear-gradient(rgba(38,48,42,.35), rgba(38,48,42,.92))"
hero_full = f"{hero_grad}, {hero_bg}" if hero_b64 else f"linear-gradient(135deg, #333F31 0%, #26302A 100%)"

# ── All custom CSS ────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Mulish:wght@300;400;600;700;800&display=swap');

:root {{
  --bg:        #26302A;
  --deep:      #333F31;
  --card:      #3B4A3D;
  --border:    #4A5B4C;
  --slate:     #445954;
  --accent:    #8D8850;
  --accent2:   #8FA2A6;
  --text:      #E7E9E4;
  --muted:     #A9B0A6;
  --low:       #95D5B2;
  --med:       #52B788;
  --high:      #C9A44C;
}}

html, body, [class*="css"], p, div, span, label, input, select, textarea {{
    font-family: 'Mulish', sans-serif !important;
    color: var(--text);
}}
h1, h2, h3, .serif {{ font-family: 'DM Serif Display', serif !important; }}

#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
header    {{ visibility: hidden; }}

.block-container {{ padding: 1.5rem 3.5rem 2rem 3.5rem; max-width: 100%; }}

/* ── Animations ── */
@keyframes pulse  {{ 0%,100%{{transform:scale(1)}} 50%{{transform:scale(1.12)}} }}
@keyframes bounce {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-7px)}} }}
@keyframes fadeIn {{ from{{opacity:0;transform:translateY(14px)}} to{{opacity:1;transform:translateY(0)}} }}

/* ── Hero ── */
.hero-banner {{
    background-image: {hero_full};
    background-size: cover;
    background-position: center;
    border-radius: 16px;
    padding: 3rem 3rem 2.6rem 3rem;
    margin-bottom: 0.5rem;
    border: 1px solid var(--border);
    box-shadow: 0 10px 36px rgba(0,0,0,0.5);
    text-align: center;
}}
.hero-emoji  {{ font-size:3.2rem; display:inline-block; animation:bounce 2.8s ease-in-out infinite; }}
.hero-title  {{ font-family:'DM Serif Display',serif !important; font-size:clamp(2rem,5vw,3.2rem); color:var(--text); margin:0.4rem 0 0.3rem; letter-spacing:-0.5px; }}
.hero-sub    {{ font-size:1rem; color:rgba(231,233,228,0.8); font-weight:400; }}
.olive-divider {{ border:none; border-top:2px solid var(--accent); margin:1.4rem 0; opacity:0.65; }}

/* ── Section headers ── */
.section-header {{
    font-family:'DM Serif Display',serif !important;
    font-size:1.15rem; font-weight:400; color:var(--text);
    padding-bottom:0.4rem; margin-bottom:1rem;
    border-bottom: 2px solid var(--accent);
    display: inline-block;
    padding-right: 2rem;
}}

/* ── Lag box ── */
.lag-box {{
    background:var(--card); border:1px solid var(--border); border-radius:12px;
    padding:0.85rem 1.1rem; font-size:0.9rem; color:var(--text);
    box-shadow:0 2px 8px rgba(0,0,0,0.25);
}}
.lag-ok {{ color:var(--low); font-weight:700; }}

/* ── Result cards ── */
.result-card {{
    background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:1.8rem 2rem; text-align:center;
    box-shadow:0 6px 22px rgba(0,0,0,0.35); animation:fadeIn 0.5s ease-out;
}}
.result-label {{
    font-size:0.68rem; font-weight:700; letter-spacing:1.5px;
    text-transform:uppercase; color:var(--muted); margin-bottom:0.8rem;
}}
.result-value-large {{
    font-family:'DM Serif Display',serif !important;
    font-size:2.8rem; line-height:1.1;
}}
.badge-low  {{ color:var(--low); }}
.badge-med  {{ color:var(--med); }}
.badge-high {{ color:var(--high); }}

/* ── Prob bars ── */
.prob-row   {{ display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem; font-size:0.88rem; }}
.prob-label {{ width:56px; color:var(--muted); font-weight:600; }}
.prob-bar-bg  {{ flex:1; background:var(--deep); border-radius:6px; height:9px; overflow:hidden; }}
.prob-bar-fill {{ height:100%; border-radius:6px; }}
.prob-pct   {{ width:36px; text-align:right; color:var(--text); font-weight:700; }}

/* ── Comparison cards ── */
.compare-card {{
    background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:1.5rem 1.8rem; text-align:center;
    box-shadow:0 6px 22px rgba(0,0,0,0.35); animation:fadeIn 0.5s ease-out;
}}
.compare-country {{ font-family:'DM Serif Display',serif !important; font-size:1.2rem; color:var(--accent); margin-bottom:0.5rem; }}
.compare-level   {{ font-family:'DM Serif Display',serif !important; font-size:2.2rem; margin-bottom:0.3rem; }}
.compare-nights  {{ font-size:1rem; color:var(--low); font-weight:600; }}

/* ── Chart wrapper ── */
.chart-card {{
    background:var(--deep); border:1px solid var(--border); border-radius:14px;
    padding:1.2rem 1.4rem; box-shadow:0 4px 16px rgba(0,0,0,0.3); margin-bottom:0.5rem;
}}

/* ── Campsite cards ── */
.camp-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:1rem; margin-top:0.8rem; }}
.camp-card {{
    background:var(--card); border:1px solid var(--border); border-top:3px solid var(--accent);
    border-radius:14px; padding:1rem 1.2rem;
    box-shadow:0 4px 14px rgba(0,0,0,0.3); animation:fadeIn 0.4s ease-out;
}}
.camp-name  {{ font-family:'DM Serif Display',serif !important; font-size:1rem; color:var(--text); margin-bottom:0.3rem; }}
.camp-stars {{ color:var(--high); font-size:0.9rem; margin-bottom:0.2rem; }}
.camp-price {{ font-size:0.8rem; color:var(--low); font-weight:600; margin-bottom:0.6rem; }}
.camp-link  {{
    display:inline-block; font-size:0.78rem; font-weight:700;
    color:var(--bg); background:var(--accent); border-radius:12px;
    padding:0.32rem 0.85rem; text-decoration:none;
    box-shadow:0 2px 8px rgba(141,136,80,0.3);
}}
.camp-link:hover {{ background:#7a7644; }}

/* ── Buttons ── */
div.stButton > button {{
    font-family:'Mulish',sans-serif !important; font-weight:700; font-size:1.05rem;
    border-radius:12px; padding:0.65rem 1.5rem;
    background:var(--accent); color:var(--bg); border:none;
    box-shadow:0 4px 14px rgba(141,136,80,0.3);
    transition:background 0.2s, box-shadow 0.2s, transform 0.1s;
}}
div.stButton > button:hover {{
    background:#7a7644; box-shadow:0 6px 20px rgba(141,136,80,0.45); transform:translateY(-1px);
}}

/* ── Footer ── */
.app-footer {{
    text-align:center; font-size:0.78rem; color:var(--text);
    margin-top:2.5rem; padding-top:1rem;
    border-top:2px solid var(--accent); opacity:0.7;
}}

/* ── Mobile ── */
@media (max-width:768px) {{
    .block-container {{ padding:1rem; }}
    .hero-title {{ font-size:1.8rem; }}
    .hero-banner {{ padding:2rem 1.4rem; }}
    .result-value-large {{ font-size:2rem; }}
    .compare-level {{ font-size:1.6rem; }}
}}
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
PROB_COLORS = ["#95D5B2", "#52B788", "#C9A44C"]

# ── Curated campsite data ─────────────────────────────────────────────────────
TOP_CAMPSITES = {
    "FR": [
        {"name":"Camping Le Vieux Port",         "lat":43.813,"lon":-1.393,"stars":"⭐⭐⭐⭐⭐","price":"from €30/night","url":"https://www.camping-le-vieux-port.com"},
        {"name":"Yelloh! Village Le Brasilia",    "lat":42.705,"lon": 3.037,"stars":"⭐⭐⭐⭐⭐","price":"from €28/night","url":"https://www.lebrasilia.fr"},
        {"name":"Camping Les Sablons",            "lat":43.421,"lon": 3.537,"stars":"⭐⭐⭐⭐⭐","price":"from €25/night","url":"https://www.les-sablons.com"},
        {"name":"Camping La Côte Sauvage",        "lat":45.343,"lon":-1.142,"stars":"⭐⭐⭐⭐","price":"from €22/night","url":"https://www.camping-lacotesauvage.com"},
        {"name":"Sandaya Domaine de la Forêt",    "lat":47.653,"lon":-2.428,"stars":"⭐⭐⭐⭐⭐","price":"from €35/night","url":"https://www.sandaya.fr"},
    ],
    "DE": [
        {"name":"Camping Park Sanssouci Potsdam", "lat":52.396,"lon":13.017,"stars":"⭐⭐⭐⭐","price":"from €20/night","url":"https://www.camping-sanssouci.de"},
        {"name":"Camping Waldsee Freiburg",       "lat":47.985,"lon": 7.893,"stars":"⭐⭐⭐⭐","price":"from €25/night","url":"https://www.camping-freiburg.com"},
        {"name":"Campingplatz Thalkirchen München","lat":48.085,"lon":11.538,"stars":"⭐⭐⭐⭐","price":"from €22/night","url":"https://www.campingplatz-muenchen.de"},
        {"name":"Weissenhäuser Strand Camping",   "lat":54.342,"lon":10.862,"stars":"⭐⭐⭐⭐⭐","price":"from €30/night","url":"https://www.weissenhaeuser-strand.de"},
    ],
    "IT": [
        {"name":"Camping Village Cavallino",      "lat":45.468,"lon":12.523,"stars":"⭐⭐⭐⭐⭐","price":"from €35/night","url":"https://www.villaggiocavallino.it"},
        {"name":"Camping Cisano San Vito",        "lat":45.601,"lon":10.724,"stars":"⭐⭐⭐⭐⭐","price":"from €28/night","url":"https://www.camping-cisano.it"},
        {"name":"Camping Tahiti Lido di Jesolo",  "lat":45.512,"lon":12.627,"stars":"⭐⭐⭐⭐⭐","price":"from €32/night","url":"https://www.campingtahiti.com"},
        {"name":"Camping Internazionale Firenze", "lat":43.783,"lon":11.278,"stars":"⭐⭐⭐⭐","price":"from €22/night","url":"https://www.campingfirenze.com"},
    ],
    "ES": [
        {"name":"Camping Playa Montroig",         "lat":40.978,"lon": 0.960,"stars":"⭐⭐⭐⭐⭐","price":"from €30/night","url":"https://www.playamontroig.com"},
        {"name":"Camping Las Dunas Sant Pere",    "lat":42.193,"lon": 3.099,"stars":"⭐⭐⭐⭐⭐","price":"from €35/night","url":"https://www.campinglasdunas.com"},
        {"name":"Camping El Garrofer Sitges",     "lat":41.217,"lon": 1.817,"stars":"⭐⭐⭐⭐","price":"from €25/night","url":"https://www.elgarrofer.com"},
    ],
    "HR": [
        {"name":"Camping Zaton Holiday Resort",   "lat":44.213,"lon":15.172,"stars":"⭐⭐⭐⭐⭐","price":"from €25/night","url":"https://www.zaton.hr"},
        {"name":"Camping Solaris Šibenik",        "lat":43.723,"lon":15.874,"stars":"⭐⭐⭐⭐⭐","price":"from €30/night","url":"https://www.solarisbeachresort.hr"},
        {"name":"Camping Straško Novalja",        "lat":44.552,"lon":14.889,"stars":"⭐⭐⭐⭐","price":"from €22/night","url":"https://www.strasko.hr"},
    ],
    "AT": [
        {"name":"Camping Wolfgangsee",            "lat":47.739,"lon":13.443,"stars":"⭐⭐⭐⭐⭐","price":"from €28/night","url":"https://www.camping-wolfgangsee.at"},
        {"name":"Camping Innsbruck Kranebitten",  "lat":47.267,"lon":11.323,"stars":"⭐⭐⭐⭐","price":"from €20/night","url":"https://www.camping-innsbruck.at"},
        {"name":"Camping Neue Donau Wien",        "lat":48.228,"lon":16.432,"stars":"⭐⭐⭐⭐","price":"from €22/night","url":"https://www.campingwien.at"},
    ],
    "NL": [
        {"name":"Camping De Lievelinge",          "lat":51.787,"lon": 5.083,"stars":"⭐⭐⭐⭐⭐","price":"from €22/night","url":"https://www.delievelinge.nl"},
        {"name":"Camping Koningshof",             "lat":52.243,"lon": 4.539,"stars":"⭐⭐⭐⭐⭐","price":"from €25/night","url":"https://www.koningshof.nl"},
        {"name":"Camping De Groote Vliet",        "lat":51.643,"lon": 4.503,"stars":"⭐⭐⭐⭐","price":"from €20/night","url":"https://www.grootevliet.nl"},
    ],
    "PT": [
        {"name":"Camping Orbitur Guincho",        "lat":38.731,"lon":-9.469,"stars":"⭐⭐⭐⭐⭐","price":"from €25/night","url":"https://www.orbitur.pt"},
        {"name":"Camping Sagres",                 "lat":37.012,"lon":-8.941,"stars":"⭐⭐⭐⭐","price":"from €15/night","url":"https://www.campingsagres.pt"},
        {"name":"Camping Viana do Castelo",       "lat":41.691,"lon":-8.831,"stars":"⭐⭐⭐⭐","price":"from €18/night","url":"https://www.campingviana.com.pt"},
    ],
    "SE": [
        {"name":"First Camp Djurgården Stockholm","lat":59.341,"lon":18.109,"stars":"⭐⭐⭐⭐⭐","price":"from €35/night","url":"https://www.firstcamp.se"},
        {"name":"Camping Kolmården",              "lat":58.672,"lon":16.378,"stars":"⭐⭐⭐⭐","price":"from €25/night","url":"https://www.kolmarden.com"},
    ],
    "NO": [
        {"name":"Camping Geiranger Storfjord",    "lat":62.101,"lon": 7.205,"stars":"⭐⭐⭐⭐⭐","price":"from €30/night","url":"https://www.geirangerfjord.no"},
        {"name":"Camping Preikestolen Stavanger", "lat":58.987,"lon": 5.978,"stars":"⭐⭐⭐⭐","price":"from €25/night","url":"https://www.preikestolencamping.com"},
    ],
    "DK": [
        {"name":"Camping Blåvand Strand",         "lat":55.562,"lon": 8.098,"stars":"⭐⭐⭐⭐⭐","price":"from €28/night","url":"https://www.blaavandstrand.dk"},
        {"name":"Camping Hesselet Nyborg",        "lat":55.312,"lon":10.818,"stars":"⭐⭐⭐⭐⭐","price":"from €25/night","url":"https://www.hesselet.dk"},
    ],
    "CH": [
        {"name":"Camping Romantik Interlaken",    "lat":46.684,"lon": 7.868,"stars":"⭐⭐⭐⭐⭐","price":"from €35/night","url":"https://www.camping-interlaken.ch"},
        {"name":"Camping Zürich",                 "lat":47.388,"lon": 8.497,"stars":"⭐⭐⭐⭐","price":"from €28/night","url":"https://www.camping-zurich.ch"},
    ],
    "PL": [
        {"name":"Camping Zakopane",               "lat":49.295,"lon":19.958,"stars":"⭐⭐⭐⭐","price":"from €15/night","url":"https://www.camping-zakopane.pl"},
        {"name":"Camping Marina Gdańsk",          "lat":54.406,"lon":18.619,"stars":"⭐⭐⭐⭐","price":"from €12/night","url":"https://www.camping-gdansk.pl"},
    ],
    "EL": [
        {"name":"Camping Ionion Beach Lefkada",   "lat":38.718,"lon":20.642,"stars":"⭐⭐⭐⭐","price":"from €20/night","url":"https://www.ionionbeach.gr"},
        {"name":"Camping Plaka Naxos",            "lat":37.038,"lon":25.508,"stars":"⭐⭐⭐⭐","price":"from €18/night","url":"https://www.campingplaka.gr"},
    ],
    "BE": [
        {"name":"Camping Floréal La Roche",       "lat":50.183,"lon": 5.576,"stars":"⭐⭐⭐⭐⭐","price":"from €25/night","url":"https://www.florealgroup.be"},
        {"name":"Camping Grimbergen Brussels",    "lat":50.935,"lon": 4.365,"stars":"⭐⭐⭐⭐","price":"from €20/night","url":"https://www.campinggrimbergen.be"},
    ],
    "CZ": [
        {"name":"Camping Yacht Club Praha",       "lat":50.043,"lon":14.408,"stars":"⭐⭐⭐⭐","price":"from €15/night","url":"https://www.campingprague.cz"},
        {"name":"Autocamp Třeboň",                "lat":49.005,"lon":14.771,"stars":"⭐⭐⭐⭐","price":"from €12/night","url":"https://www.autocamp-trebon.cz"},
    ],
    "HU": [
        {"name":"Camping Balatontourist Füred",   "lat":46.958,"lon":17.898,"stars":"⭐⭐⭐⭐","price":"from €18/night","url":"https://www.balatontourist.hu"},
        {"name":"Camping Jumbo Budapest",         "lat":47.499,"lon":19.071,"stars":"⭐⭐⭐","price":"from €14/night","url":"https://www.campingjumbo.hu"},
    ],
    "SI": [
        {"name":"Camping Bled",                   "lat":46.368,"lon":14.093,"stars":"⭐⭐⭐⭐⭐","price":"from €25/night","url":"https://www.camping-bled.com"},
        {"name":"Camping Kolpa Metlika",          "lat":45.646,"lon":15.319,"stars":"⭐⭐⭐⭐","price":"from €18/night","url":"https://www.campingkolpa.si"},
    ],
    "SK": [
        {"name":"Camping Zlaté Piesky Bratislava","lat":48.178,"lon":17.178,"stars":"⭐⭐⭐⭐","price":"from €15/night","url":"https://www.zlatepiesky.sk"},
        {"name":"Camping Liptov",                 "lat":49.085,"lon":19.601,"stars":"⭐⭐⭐⭐","price":"from €18/night","url":"https://www.campliptov.sk"},
    ],
    "RO": [
        {"name":"Camping La Conac Transylvania",  "lat":45.648,"lon":25.612,"stars":"⭐⭐⭐⭐","price":"from €12/night","url":"https://www.laconac.ro"},
        {"name":"Camping Neptun Black Sea",       "lat":43.895,"lon":28.592,"stars":"⭐⭐⭐","price":"from €10/night","url":"https://www.camping-neptun.ro"},
    ],
    "BG": [
        {"name":"Camping Gradina Sozopol",        "lat":42.381,"lon":27.699,"stars":"⭐⭐⭐⭐","price":"from €12/night","url":"https://www.campingbg.com"},
    ],
    "FI": [
        {"name":"Camping Rastila Helsinki",       "lat":60.199,"lon":25.117,"stars":"⭐⭐⭐⭐","price":"from €25/night","url":"https://www.campingrastila.fi"},
        {"name":"Camping Naantali",               "lat":60.469,"lon":22.017,"stars":"⭐⭐⭐⭐","price":"from €22/night","url":"https://www.campingnaantali.fi"},
    ],
    "IE": [
        {"name":"Cong Camping & Caravan Park",    "lat":53.536,"lon":-9.279,"stars":"⭐⭐⭐⭐","price":"from €20/night","url":"https://www.cong-camping.com"},
        {"name":"Camac Valley Tourist Park Dublin","lat":53.313,"lon":-6.358,"stars":"⭐⭐⭐⭐","price":"from €25/night","url":"https://www.camacvalley.ie"},
    ],
    "UK": [
        {"name":"Camping Longleat Wiltshire",     "lat":51.187,"lon":-2.278,"stars":"⭐⭐⭐⭐⭐","price":"from €30/night","url":"https://www.longleat.co.uk"},
        {"name":"Camping Loch Lomond",            "lat":56.173,"lon":-4.601,"stars":"⭐⭐⭐⭐","price":"from €25/night","url":"https://www.lochlomond.com"},
    ],
    "LU": [
        {"name":"Camping Echternach",             "lat":49.814,"lon": 6.417,"stars":"⭐⭐⭐⭐⭐","price":"from €22/night","url":"https://www.camping-echternach.lu"},
        {"name":"Camping Kockelscheuer",          "lat":49.566,"lon": 6.113,"stars":"⭐⭐⭐⭐","price":"from €20/night","url":"https://www.camping-kockelscheuer.lu"},
    ],
    "IS": [
        {"name":"Camping Þórsmörk",               "lat":63.681,"lon":-19.511,"stars":"⭐⭐⭐⭐⭐","price":"from €18/night","url":"https://www.utivist.is"},
        {"name":"Camping Reykjavík",              "lat":64.129,"lon":-21.894,"stars":"⭐⭐⭐⭐","price":"from €20/night","url":"https://www.campingreykjavik.is"},
    ],
    "EE": [
        {"name":"Camping Pirita Tallinn",         "lat":59.462,"lon":24.839,"stars":"⭐⭐⭐⭐","price":"from €20/night","url":"https://www.piritacamping.ee"},
        {"name":"Camping Pärnu Beach",            "lat":58.371,"lon":24.509,"stars":"⭐⭐⭐⭐","price":"from €18/night","url":"https://www.parnu.ee"},
    ],
    "LT": [
        {"name":"Camping Nida Curonian Spit",     "lat":55.307,"lon":21.005,"stars":"⭐⭐⭐⭐⭐","price":"from €20/night","url":"https://www.neringa.lt"},
    ],
    "LV": [
        {"name":"Camping Jūrmala",                "lat":56.969,"lon":23.683,"stars":"⭐⭐⭐⭐","price":"from €18/night","url":"https://www.jurmala.lv"},
    ],
    "TR": [
        {"name":"Camping Olympos Antalya",        "lat":36.402,"lon":30.469,"stars":"⭐⭐⭐⭐","price":"from €12/night","url":"https://www.olymposcamp.com"},
        {"name":"Camping Ölüdeniz",               "lat":36.549,"lon":29.115,"stars":"⭐⭐⭐⭐","price":"from €10/night","url":"https://www.oludeniz.com"},
    ],
    "ME": [
        {"name":"Camping Full Monte Kotor",       "lat":42.424,"lon":18.771,"stars":"⭐⭐⭐⭐","price":"from €15/night","url":"https://www.fullmonte.me"},
    ],
    "MT": [
        {"name":"Mellieħa Holiday Centre",        "lat":35.962,"lon":14.362,"stars":"⭐⭐⭐⭐","price":"from €18/night","url":"https://www.melliehaholidaycentre.com"},
    ],
    "CY": [
        {"name":"Camping Forest Park Troodos",    "lat":34.921,"lon":32.878,"stars":"⭐⭐⭐","price":"from €12/night","url":"https://www.cypruscamp.com"},
    ],
}

# ── Feature builder ───────────────────────────────────────────────────────────
def lookup_lag(geo, yr, mo, shift):
    mo2, yr2 = mo - shift, yr
    while mo2 < 1:
        mo2 += 12; yr2 -= 1
    row = hist[(hist["geo"] == geo) & (hist["year"] == yr2) & (hist["month"] == mo2)]
    return float(row["nights_spent"].values[0]) if len(row) else None

def build_features(geo, yr, mo):
    l1  = lookup_lag(geo, yr, mo, 1)  or 0
    l12 = lookup_lag(geo, yr, mo, 12) or 0
    return pd.DataFrame([{
        "geo": geo, "year_rel": yr - 2010, "month": mo,
        "month_sin": np.sin(2 * np.pi * mo / 12),
        "month_cos": np.cos(2 * np.pi * mo / 12),
        "lag_1m": l1, "lag_12m": l12,
        "is_summer": int(mo in [6,7,8]),
        "is_july":   int(mo == 7),
        "is_covid":  int(yr in [2020,2021]),
    }])

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-emoji">⛺</div>
  <div class="hero-title">EU Campsite Demand Forecaster</div>
  <div class="hero-sub">Predict demand level and overnight stays for any European country and month.</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr class="olive-divider">', unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⛺ Select Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("FR"))
with col2:
    year = st.number_input("Year", min_value=2010, max_value=2030, value=2025)
with col3:
    month = st.selectbox("Month", range(1, 13), format_func=lambda m: MONTH_NAMES[m-1], index=6)

st.markdown('<hr class="olive-divider">', unsafe_allow_html=True)

# ── Lag features ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Lag Features</div>', unsafe_allow_html=True)

lag_1m_auto  = lookup_lag(country, year, month, 1)
lag_12m_auto = lookup_lag(country, year, month, 12)

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

st.markdown('<hr class="olive-divider">', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
_, bcol, _ = st.columns([1, 2, 1])
with bcol:
    predict = st.button("Predict 🔍", type="primary", use_container_width=True)

if predict:
    X = build_features(country, year, month)
    X.at[0, "lag_1m"]  = lag_1m
    X.at[0, "lag_12m"] = lag_12m

    demand_code  = int(clf_model.predict(X)[0])
    demand_proba = clf_model.predict_proba(X)[0]
    nights_pred  = max(0, float(reg_model.predict(X)[0]))
    labels       = ["Low", "Medium", "High"]
    badge_class  = ["badge-low", "badge-med", "badge-high"]

    avg    = hist[(hist["geo"] == country) & (hist["month"] == month)]["nights_spent"].mean()
    interp = ("Above average" if nights_pred > avg * 1.1
              else "Below average" if nights_pred < avg * 0.9
              else "Around average")
    interp += f" for {country} in {MONTH_NAMES[month-1]}"

    st.markdown('<hr class="olive-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔍 Results</div>', unsafe_allow_html=True)

    rc1, _, rc3 = st.columns([2, 0.15, 2])
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
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{pct}%;background:{PROB_COLORS[i]};"></div></div>
                <span class="prob-pct">{pct}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with rc3:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Nights Spent</div>
            <div class="result-value-large" style="color:var(--high);">⛺ {nights_pred:,.0f}</div>
            <div style="margin-top:0.8rem;font-size:0.88rem;color:var(--low);font-weight:600;">{interp}</div>
            <div style="margin-top:0.4rem;font-size:0.78rem;color:var(--muted);">{country} · {MONTH_NAMES[month-1]} {year}</div>
        </div>""", unsafe_allow_html=True)

# ── Historical chart ──────────────────────────────────────────────────────────
st.markdown('<hr class="olive-divider">', unsafe_allow_html=True)
st.markdown(f'<div class="section-header">📊 Historical Trend — {country}</div>', unsafe_allow_html=True)

country_hist = hist[hist["geo"] == country].copy()
country_hist["date"] = pd.to_datetime(country_hist[["year","month"]].assign(day=1))
country_hist = country_hist.sort_values("date")

if len(country_hist) > 0:
    fig, ax = plt.subplots(figsize=(14, 3.4))
    fig.patch.set_facecolor("#333F31")
    ax.set_facecolor("#333F31")
    ax.plot(country_hist["date"], country_hist["nights_spent"] / 1e6,
            color="#C9A44C", linewidth=1.8, zorder=3)
    ax.fill_between(country_hist["date"], country_hist["nights_spent"] / 1e6,
                    alpha=0.15, color="#C9A44C")
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
               alpha=0.12, color="#a05050", zorder=1)
    ax.text(pd.Timestamp("2020-06-01"), ax.get_ylim()[1] * 0.92,
            "COVID-19", fontsize=7.5, color="#c07070", va="top")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.set_ylabel("Nights spent", fontsize=8.5, color="#A9B0A6")
    ax.tick_params(colors="#A9B0A6", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#4A5B4C", linewidth=0.8)
    plt.tight_layout()
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()

# ── Country comparison ────────────────────────────────────────────────────────
st.markdown('<hr class="olive-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🏆 Country Comparison</div>', unsafe_allow_html=True)

other_default = COUNTRIES.index("DE") if country != "DE" else COUNTRIES.index("FR")
other_country = st.selectbox("Compare with", COUNTRIES, index=other_default, key="compare_country")

if st.button("Compare 🔍", use_container_width=False):
    labels      = ["Low", "Medium", "High"]
    badge_class = ["badge-low", "badge-med", "badge-high"]
    badge_emoji = ["🟢", "🟡", "🔴"]

    X1 = build_features(country,       year, month)
    X2 = build_features(other_country, year, month)
    code1   = int(clf_model.predict(X1)[0])
    nights1 = max(0, float(reg_model.predict(X1)[0]))
    code2   = int(clf_model.predict(X2)[0])
    nights2 = max(0, float(reg_model.predict(X2)[0]))

    cc1, _, cc2 = st.columns([2, 0.2, 2])
    with cc1:
        st.markdown(f"""
        <div class="compare-card" style="border-top:3px solid #C9A44C;">
            <div class="compare-country">{country}</div>
            <div class="compare-level {badge_class[code1]}">{badge_emoji[code1]} {labels[code1]}</div>
            <div class="compare-nights">⛺ {nights1:,.0f} nights</div>
            <div style="font-size:0.78rem;color:var(--muted);margin-top:0.4rem;">{MONTH_NAMES[month-1]} {year}</div>
        </div>""", unsafe_allow_html=True)
    with cc2:
        st.markdown(f"""
        <div class="compare-card" style="border-top:3px solid #8FA2A6;">
            <div class="compare-country" style="color:#8FA2A6;">{other_country}</div>
            <div class="compare-level {badge_class[code2]}">{badge_emoji[code2]} {labels[code2]}</div>
            <div class="compare-nights">⛺ {nights2:,.0f} nights</div>
            <div style="font-size:0.78rem;color:var(--muted);margin-top:0.4rem;">{MONTH_NAMES[month-1]} {year}</div>
        </div>""", unsafe_allow_html=True)

    h1 = hist[hist["geo"] == country].copy()
    h2 = hist[hist["geo"] == other_country].copy()
    for h in [h1, h2]:
        h["date"] = pd.to_datetime(h[["year","month"]].assign(day=1))

    fig2, ax2 = plt.subplots(figsize=(14, 3.2))
    fig2.patch.set_facecolor("#333F31")
    ax2.set_facecolor("#333F31")
    ax2.plot(h1.sort_values("date")["date"], h1.sort_values("date")["nights_spent"] / 1e6,
             color="#C9A44C", linewidth=1.6, label=country)
    ax2.fill_between(h1.sort_values("date")["date"], h1.sort_values("date")["nights_spent"] / 1e6,
                     alpha=0.1, color="#C9A44C")
    ax2.plot(h2.sort_values("date")["date"], h2.sort_values("date")["nights_spent"] / 1e6,
             color="#8FA2A6", linewidth=1.6, label=other_country)
    ax2.fill_between(h2.sort_values("date")["date"], h2.sort_values("date")["nights_spent"] / 1e6,
                     alpha=0.1, color="#8FA2A6")
    ax2.legend(facecolor="#333F31", edgecolor="#4A5B4C", labelcolor="#E7E9E4", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax2.set_ylabel("Nights spent", fontsize=8.5, color="#A9B0A6")
    ax2.tick_params(colors="#A9B0A6", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(axis="y", color="#4A5B4C", linewidth=0.8)
    plt.tight_layout()
    st.markdown('<div class="chart-card" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.pyplot(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()

# ── Map + campsites ───────────────────────────────────────────────────────────
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

st.markdown('<hr class="olive-divider">', unsafe_allow_html=True)
st.markdown(f'<div class="section-header">🗺️ Top Campsites in {country}</div>', unsafe_allow_html=True)

lat, lon, zoom = COUNTRY_COORDS.get(country, (54.0, 15.0, 4))
campsites = TOP_CAMPSITES.get(country, [])

m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="CartoDB dark_matter",
               zoom_control=True, scrollWheelZoom=False)
folium.Marker(location=[lat, lon], tooltip=country,
              icon=folium.Icon(color="orange", icon="star")).add_to(m)
for cs in campsites:
    folium.Marker(
        location=[cs["lat"], cs["lon"]],
        tooltip=cs["name"],
        popup=folium.Popup(
            f"<b>{cs['name']}</b><br>{cs['stars']}<br>{cs['price']}<br>"
            f"<a href='{cs['url']}' target='_blank'>Visit website →</a>",
            max_width=220,
        ),
        icon=folium.Icon(color="green", icon="home"),
    ).add_to(m)

st_folium(m, use_container_width=True, height=420, returned_objects=[])

if campsites:
    cards_html = '<div class="camp-grid">'
    for cs in campsites:
        cards_html += f"""
        <div class="camp-card">
            <div class="camp-name">{cs['name']}</div>
            <div class="camp-stars">{cs['stars']}</div>
            <div class="camp-price">💶 {cs['price']}</div>
            <a class="camp-link" href="{cs['url']}" target="_blank">Visit website →</a>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
else:
    st.markdown('<p style="color:var(--muted);font-size:0.9rem;">No curated campsites yet for this country.</p>',
                unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">Built with ⛺ by Nuša Brezovnik Bunderla &nbsp;·&nbsp; Data: Eurostat CC BY 4.0</div>',
    unsafe_allow_html=True
)
