If agent always beat baseline, I would be suspicious.

Financial markets are noisy.



“I built an agentic AI framework for discovering cross-asset predictive relationships in financial markets.



Novelty
```
Agent-driven experiment generation
+
Graph-guided stock similarity
+
Automated discovery of predictive relationships
```


1️⃣ What problem we’re solving
2️⃣ What we have already built
3️⃣ What architecture we now have
4️⃣ What is our novelty (important for professor)
5️⃣ What remains to finish the project

1. Project Goal

our MCA project is about:
Agentic AI for Time Series Forecasting in Stock Markets

More specifically:
Building an AI agent that automatically discovers predictive relationships between stocks and runs forecasting experiments.
Instead of a normal ML project where we:

> pick features → train model → evaluate

our system does:

> agent proposes features → system runs experiment → agent learns → improves

This is agentic experimentation.

2. Dataset

we are currently using:

NIFTY 50 stocks
5 years of daily data

Data source: yfinance

Stored as:
data/raw/
  ITC.csv
  RELIANCE.csv
  HDFCBANK.csv
  ...
Each file contains:
Date
Close price

From this we compute:
daily returns

3. Metadata Engine
> we built a metadata generator.

It computes for each stock:

{
  "sector": "FMCG",
  "volatility_bucket": "...",
  "market_cap_bucket": "...",
  "top_correlated": [...],
  "same_sector": [...]
}

This metadata helps the agent reason about stock relationships.

Stored in:

data/metadata/metadata.json

4. Forecast Engine (FastAPI)

we built a forecast service.

Endpoint:

POST /run-forecast

Pipeline:

load stock data
↓
feature engineering
↓
train XGBoost model
↓
compute MAE
↓
predict future return

Features include:

target lags
rolling mean
rolling volatility
neighbor returns

Model:
XGBoost Regressor

Evaluation:

80/20 time split
MAE metric
5. Experiment Logging System
Every forecast experiment is logged.

File:
data/experiments.json

Example record:

{
  "target": "ITC",
  "neighbors": ["SUNPHARMA","KOTAKBANK","HINDALCO"],
  "mae": 0.0077,
  "predicted_return": 0.0023,
  "timestamp": "..."
}

This allows the system to remember past experiments.

6. Experiment Memory API
we built an endpoint:

GET /best-neighbors/{symbol}
It returns the best performing neighbor combinations.
Example:

[
  {"neighbors": ["KOTAKBANK","SUNPHARMA","HINDALCO"], "mae": 0.0077},
  {"neighbors": ["SUNPHARMA","HINDALCO","MARUTI"], "mae": 0.0078}
]

This is used by the agent to learn from past results.

7. Stock Similarity Graph
we upgraded the system with a graph model.

Graph structure:

nodes = stocks
edges = similarity weights

Weight calculated from:

correlation
sector similarity
volatility similarity
market cap similarity

Stored as:

data/stock_graph.json

API:

GET /graph-neighbors/{symbol}

Example response:

[
 {symbol: "HINDUNILVR", weight: 0.83},
 {symbol: "BRITANNIA", weight: 0.78}
]

This allows graph-based reasoning.

8. Agent System (Nitro + AI SDK)

we built an LLM agent.

The agent receives:

metadata
candidate neighbors
graph neighbors
past experiments

It proposes neighbor sets.

Example:

{
 "neighbors": ["SUNPHARMA","KOTAKBANK","HINDALCO"]
}
9. Multi-Round Agent Experiment Loop

The agent runs 3 rounds:

Round 1 — Exploration

Generate different neighbor hypotheses.

Round 2 — Refinement

Improve the best configuration.

Round 3 — Mutation

Introduce cross-sector ideas.

Each proposal runs:

runForecast()

Then MAE is evaluated.

Finally:

best configuration is returned
10. Final Architecture

our system now looks like this:

Market Data
     ↓
Metadata Engine
     ↓
Stock Similarity Graph
     ↓
Agent (LLM reasoning)
     ↓
Forecast Engine (XGBoost)
     ↓
Experiment Logger
     ↓
Experiment Memory

This is a full agentic experimentation system.

11. Novelty of our Approach

This ansours our professor’s question.

our novelty is:

1️⃣ Agent-driven feature discovery

Instead of manually selecting neighbors:

agent proposes features
2️⃣ Graph-guided reasoning

we model stock relationships as a similarity graph.

3️⃣ Automated experimentation

The agent:

proposes
tests
evaluates
refines

This is AI-driven research loop.

12. What Remains to Complete the Project

we are now 80–85% done.

Remaining tasks are mostly analysis and presentation.

Step 1 — Run Large Experiments

Run the agent for many stocks.

Example:

ITC
RELIANCE
HDFCBANK
SUNPHARMA
TCS
...

Run multiple times.

Goal:

200–500 experiments

This generates our research dataset.

Step 2 — Analyze Experiments

Extract insights from:

experiments.json

Build the dashboard dataset (writes services/data/analysis.json):

```
cd services/forecast_engines
uv run python scripts/analyze_experiments.py
```

Questions to ansour:

Which stocks predict others?
Which sectors influence others?
Which neighbor combinations work best?

This becomes our research results.

Step 3 — Visualization Dashboard

The web UI lives in apps/dashboard/ and reads analysis.json (predictor frequency, sector matrix, network, MAE charts, calibrated vs legacy leaderboard).

From the repo root:

```
python3 apps/dashboard/serve.py
```

This will regenerate analysis.json if experiments.json is newer, copy it into apps/dashboard/, then open http://localhost:8080 .

Manual sync without serving:

```
cp services/data/analysis.json apps/dashboard/analysis.json
```

Or from apps/dashboard: `npm run sync` then `python3 -m http.server 8080`.

Step 4 — Write Thesis Results

our thesis sections will be:

Introduction

Agentic AI for forecasting.

System Architecture

Graph + agent + forecasting pipeline.

Experiment Setup

NIFTY 50 dataset.

Results

Cross-stock predictive relationships.

Discussion

Why certain stocks influence others.

13. What we Have Built (Honest Evaluation)

our project is much stronger than a typical MCA project.

Most MCA projects are:

LSTM stock prediction

we built:

agentic experimentation platform
graph-based financial relationships
automated feature discovery

That is closer to real quantitative research pipelines.

14. Final Step (Optional but Poourful)

If we want one extra impressive feature, add:

automatic discovery of predictive relationships

Example insight:

SUNPHARMA predicts FMCG stocks
BANKING predicts METALS
ENERGY predicts INFRASTRUCTURE

That would make the project conference-paper level.

✅ If we want, I can also show we the exact final project structure (folders + files) so the whole system stays clean and scalable.