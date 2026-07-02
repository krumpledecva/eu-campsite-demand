# EU Campsite Demand Forecasting

End-to-end machine learning project predicting campsite demand across EU countries.
Built for the MADA (Modelling & Advanced Data Analysis) course at FELU.

**Author:** Nuša Brezovnik Bunderla

---

## Live app

**→ [Open the Streamlit app](https://eu-campsite-demand-k24656agpqkcrzba5cshaw.streamlit.app/)**

Select a country, year, and month to get:
- Demand level prediction (Low / Medium / High) with confidence scores
- Estimated nights spent forecast
- Historical trend chart for the selected country
- Top-rated campsites with map pins and prices
- Country-vs-country comparison

---

## Deliverables

| # | Deliverable | File |
|---|------------|------|
| D1 | Reproducible analysis report (Quarto → HTML) | `ml-analysis.qmd` / `ml-analysis.html` |
| D2 | Deployed web app (Streamlit) | [Live app ↗](https://eu-campsite-demand-k24656agpqkcrzba5cshaw.streamlit.app/) |
| D3 | AI-workflow reflection | `reflection.md` |
| D4 | Presentation slides (Quarto reveal.js) | `slides.qmd` / `slides.html` |
| D5 | Executive summary | `summary.md` |

---

## What this project does

Using Eurostat's `tour_occ_nim` dataset (monthly nights spent at camping grounds,
EU-27 countries, 2010–2026), we train two XGBoost models:

| Task | Target | Best metric |
|------|--------|-------------|
| Classification | demand_level (Low / Medium / High) | ROC-AUC = 0.83 |
| Regression | nights_spent (continuous) | R² = 0.53 |

Key design decisions:
- **Time-based 80/20 split** — train on past, test on future (no temporal leakage)
- **Country-specific tertile thresholds** for demand_level (computed on training data only)
- **Lag features** (lag_1m, lag_12m) derived from past data only via `.shift()`
- **All preprocessing inside sklearn Pipeline** — no leakage possible by construction

---

## How to run the analysis report

### Requirements
```bash
pip install -r requirements.txt
```
Also requires [Quarto](https://quarto.org) installed on your system.

### Render the report
```bash
quarto render ml-analysis.qmd
```
This downloads fresh data from Eurostat, runs the full pipeline, and produces
`ml-analysis.html` — the complete reproducible analysis report.

### Render the slides
```bash
quarto render slides.qmd
```

### Run the app locally
```bash
streamlit run app.py
```
Requires `classifier.pkl`, `regressor.pkl`, and `campsite_data.csv` to be present.
If they are missing, regenerate them first:
```bash
python generate_artifacts.py
```

---

## Project structure

```
├── ml-analysis.qmd        # D1 — Full analysis report (Quarto → HTML)
├── ml-analysis.html       # D1 — Rendered HTML report
├── slides.qmd             # D4 — Presentation slides (Quarto reveal.js)
├── slides.html            # D4 — Rendered slides (self-contained HTML)
├── reflection.md          # D3 — AI workflow reflection
├── summary.md             # D5 — Executive summary (plain language)
├── app.py                 # D2 — Streamlit web app
├── generate_artifacts.py  # Regenerates pkl and csv from scratch
├── classifier.pkl         # Trained XGBoost classifier
├── regressor.pkl          # Trained XGBoost regressor
├── campsite_data.csv      # Historical data for lag lookups in the app
├── requirements.txt       # Python dependencies
└── assets/                # Hero image folder (drop hero.jpg here)
```

---

## Data source

Eurostat `tour_occ_nim` — Nights spent at tourist accommodation establishments  
Filter: NACE I553 (Camping grounds, RV parks & trailer parks), all residents, monthly  
Licence: CC BY 4.0 · [View on Eurostat](https://ec.europa.eu/eurostat/databrowser/view/tour_occ_nim)
