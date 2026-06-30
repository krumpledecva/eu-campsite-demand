# D5 — Executive Summary

**EU Campsite Demand Forecasting**
Nuša Brezovnik Bunderla · MADA Final Project · June 2026

---

## The question

Can publicly available data predict whether European campsites will experience
low, medium, or high demand in a given month — and estimate how many overnight
stays to expect?

This question matters to campsite operators, regional tourism boards, and
local governments who make staffing, inventory, and investment decisions months
in advance, often with little more than last year's experience to guide them.

---

## What we did

We downloaded monthly campsite occupancy data for 37 European countries from
Eurostat — the EU's official statistics office — covering 2010 to 2026.
The dataset contains roughly 7,500 observations and is freely available under
an open licence.

We trained machine learning models to answer two questions simultaneously:

- **Classification:** Will demand be Low, Medium, or High this month?
- **Regression:** How many overnight stays will there be?

Models were trained on data up to 2022 and evaluated strictly on 2023–2026
data the models had never seen — simulating real-world deployment.

---

## What we found

**The models work, within limits.**

For demand level classification, the XGBoost model correctly distinguishes
a busy month from a quiet month 83% of the time (ROC-AUC = 0.83), compared
to 50% for random guessing. For night-count forecasting, the model explains
about half of the observed variation (R² = 0.53).

The single most important signal is simple: **what happened in the same month
last year**. Campsite demand is highly seasonal and highly repeatable year over
year. The second most important signal is last month's demand — momentum is real.

Summer months (June, July, August) account for 70% of all annual nights across
Europe. The model captures this reliably. The COVID-19 shock of 2020 caused a
37% drop across Europe, which the model partially anticipated through the
collapse in lag feature values.

---

## What we recommend

**Use this model as an early-warning signal, not a guarantee.**

- **For demand level planning** (staffing, bookings, inventory): the model's
  83% ROC-AUC makes it reliably useful 1–3 months ahead. A "High" prediction
  in May for July should trigger preparation; a "Low" prediction in November
  for January confirms the quiet period.

- **For budget forecasting** (rough night-count estimates): the model gives
  directional guidance. Treat the predicted number as a planning range, not a
  precise target — especially for large countries like France or Germany where
  even a 5% model error represents hundreds of thousands of nights.

- **Do not rely on the model for unprecedented events.** COVID-19 demonstrated
  that exogenous shocks can override all historical patterns instantly. The model
  has no mechanism for detecting geopolitical crises, extreme weather events,
  or sudden travel restrictions. Human judgment remains essential.

- **Smaller countries need caution.** Countries with fewer than three years of
  consistent historical data produce unreliable lag features, and model accuracy
  for those countries should be treated with extra scepticism.

---

## Bottom line

A freely available public dataset, combined with standard machine learning
techniques, can give campsite operators and tourism agencies a meaningful
head start on demand planning. The model is honest about its limits — it
explains roughly half of demand variation and reliably identifies the direction
of demand — and that is genuinely useful for operational decision-making.
