# D3 — AI-Workflow Reflection

**Author:** Nuša Brezovnik Bunderla
**Course:** Modelling & Advanced Data Analysis — Final Project

---

## Tools used

**Claude Code (claude-sonnet-4-6)** was the primary AI assistant throughout the project,
accessed via the Claude Code CLI running in an interactive terminal session.

**MCP servers used:**
- `filesystem` MCP — to read and write project files directly without leaving the
  conversation; used to create and iteratively edit `ml-analysis.qmd`, `app.py`,
  `requirements.txt`, and all supporting scripts.
- `playwright` MCP — available but not needed; the project did not require browser
  automation.

**Claude Code features used:**
- **Multi-step code generation** — Claude scaffolded the entire ML pipeline section by
  section, from data loading through model saving, writing directly into the `.qmd` file.
- **Automatic verification** — after writing each section, Claude ran the code as a
  Python script and checked the output before proceeding, catching errors (e.g. a
  matplotlib API change from `labels=` to `tick_labels=` in version 3.11) before they
  reached the final report.
- **Context persistence** — Claude maintained the full project context across all nine
  steps, remembering design decisions (e.g. why we used a time-based split rather than
  random) and carrying them consistently through the code.

---

## How I verified the AI's output

AI assistance does not remove the responsibility to understand what the code does.
I verified the output at three levels:

**1. Numeric sanity checks** — after each step I checked that outputs made intuitive
sense. The Eurostat filter returning 5,390 rows matched the proposal's estimate of
7,500–9,000 (the difference is explained by filtering to 2010+ and removing sparse
countries). The COVID drop of 36.9% in 2020 matched published tourism statistics.
The class balance of roughly equal thirds in the training set confirmed the tertile
logic was working correctly.

**2. Leakage audit** — I manually traced the data flow to confirm: (a) tertile
thresholds were computed only on `df_train`, (b) all scalers and encoders live inside
`Pipeline` objects and are never fitted on test data, (c) lag features use `.shift()`
which strictly accesses past rows within each country group.

**3. Model results cross-check** — the feature importance rankings from XGBoost
(`lag_12m` and `is_summer` as top predictors) matched what EDA showed: same-month-last-
year and summer season are the dominant signals. If the model had ranked `year_rel` or
`is_covid` highest, that would have flagged a potential leakage issue to investigate.

---

## Effort and cost estimate

| Phase | Estimated time |
|-------|---------------|
| Environment setup + data download | 30 minutes |
| EDA and cleaning | 45 minutes |
| Feature engineering + split | 30 minutes |
| Model ladder (both tasks) | 60 minutes |
| Leaderboard + evaluation | 30 minutes |
| Streamlit app + GitHub + deployment | 45 minutes |
| Written deliverables | 45 minutes |
| **Total** | **~5 hours** |

Without AI assistance, the same project would have taken an estimated 15–20 hours,
primarily due to documentation lookup, debugging, and boilerplate writing. Claude
reduced this by approximately 70%, but all decisions — the choice of time-based
splitting, the COVID flagging strategy, the honest evaluation framing — were made
by me after understanding the tradeoffs, not delegated to the AI.

---

## What I would do differently

The model's R² of 0.53 for regression is honest but modest. With more time I would
explore: (a) log-transforming `nights_spent` before regression to reduce the influence
of large-country outliers, and (b) adding a rolling 3-month average as an additional
lag feature to capture medium-term momentum. I verified the AI's suggestion to use
country-specific tertile thresholds by checking that the resulting class balance was
genuinely close to equal thirds — it was, confirming this was the right design choice.
