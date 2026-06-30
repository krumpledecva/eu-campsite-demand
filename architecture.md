# EU Campsite Demand Forecasting — Solution Architecture

## Mermaid Source

```mermaid
flowchart LR
    subgraph Offline["🧪 Offline — done once (local machine)"]
        direction TB
        DATA[("🔌 Eurostat API\ntour_occ_nim CC BY 4.0")] --> CLEAN["⚙️ Clean + Feature Engineering\nlag_1m · lag_12m · is_summer · is_july\ncountry-specific tertile thresholds"]
        CLEAN --> TRAIN_C["🤖 Train Classifier\nLogistic Reg · Decision Tree · XGBoost\nTarget: demand_level — Low / Medium / High"]
        CLEAN --> TRAIN_R["📈 Train Regressor\nLinear Reg · Decision Tree · XGBoost\nTarget: nights_spent — continuous"]
        TRAIN_C --> PKL_C["💾 classifier.pkl"]
        TRAIN_R --> PKL_R["💾 regressor.pkl"]
    end
    subgraph Runtime["☁️ Runtime — live app (deployed)"]
        direction TB
        PKL_C --> BACKEND["⚙️ FastAPI Backend\n/predict/classification\n/predict/regression"]
        PKL_R --> BACKEND
        USER(["🙋 User"]) --> FRONTEND["🖥️ Streamlit Frontend\nSelect: country · month · year"]
        FRONTEND -->|"country, month, year, lag values"| BACKEND
        BACKEND -->|"demand level + nights forecast"| FRONTEND
    end
    style Offline fill:#e8f4f8,stroke:#2196F3,stroke-width:2px
    style Runtime fill:#f0f8e8,stroke:#4CAF50,stroke-width:2px
```

## Written Explanation

Training happens entirely offline, on a local machine, as a one-time step before deployment. The Eurostat `tour_occ_nim` dataset is downloaded, cleaned, and enriched with lag features (`lag_1m`, `lag_12m`), seasonal flags (`is_summer`, `is_july`), and country-specific tertile thresholds to derive the `demand_level` target. Two models are trained — a classifier (predicting Low / Medium / High demand) and a regressor (predicting the raw number of nights spent) — and both are saved to disk as `classifier.pkl` and `regressor.pkl`. At runtime, the FastAPI backend loads those two saved files once on startup and exposes two prediction endpoints (`/predict/classification` and `/predict/regression`); no retraining ever happens in the live app. When a user selects a country, month, and year in the Streamlit frontend, the feature values are sent to the backend, which calls `model.predict()` on both models and returns the demand level label and the nights-spent forecast back to the browser.
