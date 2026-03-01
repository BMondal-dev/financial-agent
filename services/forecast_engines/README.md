# This service is responsible for:
- Downloading historical NIFTY 50 stock data
- Generating metadata (volatility + correlation)
- Providing a forecasting API (FastAPI)
This forms the Data Layer + Forecasting Module of the project.

```uv run uvicorn main:app --reload --port 8000```


## Generate Data
```uv run python scripts/generate_data.py```

- `metadata.json` Structure

Example:

```
{
  "INFY": {
    "volatility_30d": 0.018,
    "top_correlated": [
      {"symbol": "TCS", "corr": 0.91},
      {"symbol": "HCLTECH", "corr": 0.87}
    ]
  }
}
```

This metadata will later be used by the LLM agent to generate dataset enrichment proposals.






## Why Returns > Raw Price
Raw stock prices:

- Trend upward over time
- Non-stationary
- Harder for ML models

Returns:

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

Components of the Equation: <br>
$r_t$: The rate of return at period $t$. <br>
$P_t$: The price or value of the asset at the end of the current period.<br>
$P_{t-1}$: The price or value of the asset at the end of the preceding period.


> If we forecast price directly:
- Model may just learn trend
- Looks artificially good

>If we forecast returns:
- Model must learn structure
- More honest performance


## Final Forecasting Setup

### We will predict:

$$
r_{t+h}
$$

Where: <br>
$r_t$ = daily return <br>
$ℎ$ = horizon (3, 5, 10) <br>
Model Type: Single-step prediction <br>

## Model Input Feature <br>
For each training row at time t:
Features:
- target_return_lag1
- target_return_lag2
- target_return_lag3
- neighbor1_return_lag1
- neighbor2_return_lag1
- ...
- (optional later: volatility)

Target:
- return at time t + horizon


## Example

If horizon = 5: <br>
Features at day 100:
- return_99
- return_98
- return_97
- neighbor_return_99

Target:
-return_105

That’s it.

No rolling forecast.
No recursive prediction.
Simple supervised learning.


## Why We Train Each Time ??
Because the whole research idea is:
> Change dataset → measure performance → refine dataset
If we reused the same model, the experiment would be invalid.
Each dataset proposal must be trained fresh.

## Is This Expensive?
Not really.<br>
XGBoost with:<br>
- ~5 years daily data
- 80/20 split
- small feature set
- 100 trees
It will train in milliseconds.


## NEXT PHASE: EXPERIMENT ORCHESTRATION (No LLM Yet) <br>
Before we bring AI agent in, we must build:<br>
1️⃣ Baseline experiment<br>
No neighbors.

2️⃣ Static enrichment experiment<br>
Use:<br>
- Top 3 correlated <br>
- Same-sector neighbors <br>
- And compare MAE. <br>

This proves:<br>
Does enrichment even help?<br>

If enrichment doesn’t reduce MAE, agent work is pointless.