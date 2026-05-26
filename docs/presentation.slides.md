---
theme: seriph
title: "ECHO — LLM-Orchestrated Neighbor-Enriched Forecasting for NIFTY Equity Returns"
info: |
  ## ECHO: Equity Cross-stock Heuristic Orchestrator
  Final-year project presentation.
  Author: Bijit Mondal
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
fonts:
  sans: "Inter"
  serif: "Source Serif Pro"
  mono: "JetBrains Mono"
---

# ECHO

### LLM-Orchestrated Neighbor-Enriched Forecasting for NIFTY Equity Returns

<div class="text-sm opacity-80 mt-6">
Final-year Project · Bijit Mondal
</div>

<div class="abs-br m-6 text-xs opacity-60">
56 NIFTY targets · 1,175 calibrated runs at H=10 · 1,536 at H=20 · XGBoost + LSTM
</div>

<!--
SCRIPT (45–60 sec):
Good [morning/afternoon] everyone. My final-year project is called ECHO — Equity Cross-stock Heuristic Orchestrator. The one-line pitch: we built an LLM-driven agentic platform that automates a quantitative researcher's workflow for discovering which "neighbor stocks" help predict a target stock's future return on the NIFTY 50, and we evaluate every hypothesis under a strict leakage-safe protocol. In this presentation I will explain what we built, why it matters, how we measure it, and what the data actually shows across two horizons — 10 and 20 trading days.

PROBABLE Q&A:
- Q: "One line — what is ECHO?"
  A: A reproducible agentic experimentation loop that proposes, measures, logs and refines cross-stock neighbor sets for return forecasting on NIFTY.
- Q: "Is this a trading bot?"
  A: No. It is a research infrastructure for honest measurement of neighbor designs; PnL/backtesting is downstream future work.
-->

---
layout: two-cols
---

# The problem

In quant research, the bottleneck is **not** fitting a model — it's:

- **Hypothesis generation** — which stocks help predict a target?
- **Rigorous evaluation** — calendar alignment, chronological splits, baselines.
- **Institutional memory** — what worked, what failed, with rationale.

Human researchers rely on **coarse shortcuts**:

- Same sector
- Top historical correlation
- Discretionary intuition

These shortcuts miss **transient cross-sector lead-lag** and **regime shifts**.

::right::

# Why automate it

For \(N\) stocks and a neighbor set of size \(k\), candidate hypotheses per target are:

$$\binom{N-1}{k}$$

For NIFTY 50 with \(k=3\): **18,424** hypotheses **per target**.
Across 56 targets: ≈ **1 million** combinations.

**Manual search is infeasible.**
**Naive shortcuts are biased.**

ECHO turns this into a **structured, evidence-guided sequential search** with **auditable traces**.

<!--
SCRIPT:
Equity research has two costs: fitting a model is cheap; choosing what to feed the model is expensive. Practitioners typically use three shortcuts — same sector, top correlation, or intuition. These miss cross-sector lead-lag and regime shifts. And the combinatorial space is huge: for just NIFTY 50 with three neighbors per target, that's over a million hypotheses across all targets. We need an automated, disciplined way to search this space.

PROBABLE Q&A:
- Q: "Why not just use the top-correlated stock?"
  A: Our Baseline B does exactly that and it loses on most targets — we'll show 42/56 wins for the agent over fixed-correlation neighbors under XGB at H=10.
- Q: "Why is correlation alone not enough?"
  A: Correlation is symmetric and static; predictive lead-lag is asymmetric and regime-dependent.
-->

---

# Goal — stated precisely

<v-clicks>

- **Target**: a NIFTY stock (e.g. ICICIBANK).
- **Horizon \(H\)**: 10 or 20 trading days.
- **Neighbors \(\mathcal{N}\)**: a small set (≤ 3) of other stocks providing extra features.
- **Predict**: the cumulative forward return
  $$R_{t:t+H}^{(i)} = \frac{P_{t+H}^{(i)}}{P_t^{(i)}} - 1$$
- **Evaluate**: on a strict **chronological** holdout — train on the past, test on the future.
- **Optimize**: lower **MAE** on the holdout, while beating naive baselines.

</v-clicks>

<!--
SCRIPT:
The goal is to predict the H-day cumulative return of a target stock — for example, what is ICICIBANK going to do over the next 10 or 20 trading days. Features come from the target's own past plus a small set of "neighbor" stocks. The model is trained on the past and tested only on the future — never the reverse. We score everything with MAE on a chronological holdout.

PROBABLE Q&A:
- Q: "Why cumulative return, not next-day?"
  A: Multi-day cumulative returns are more economically meaningful and less dominated by intraday microstructure noise.
- Q: "Why ≤ 3 neighbors?"
  A: To keep the per-row feature width tractable and to make agent proposals comparable across rounds. Each neighbor adds two features (lag-1 + roll-5).
-->

---

# ECHO framework — four components

<div class="grid grid-cols-2 gap-4 mt-4">

<div class="border rounded-lg p-4">

### 1. Orchestrator
LLM planner over typed evidence (metadata, graph, memory, sentiment). Emits proposals in 3 stages: **exploration → refinement → mutation**.

</div>

<div class="border rounded-lg p-4">

### 2. Forecast Service
Calendar-aligned features, chronological 80/20 split with **H-row purge**, trains XGB or LSTM, reports MAE + baselines + CV.

</div>

<div class="border rounded-lg p-4">

### 3. Experiment Logger
Append-only JSON per horizon. Each entry = hypothesis + evaluation + diagnostics. File-locked for concurrent workers.

</div>

<div class="border rounded-lg p-4">

### 4. Analyzer / Dashboard
Calibrated leaderboards, sector heatmap, predictor frequency, head-to-head benchmark, stock explorer.

</div>

</div>

<div class="text-center mt-4 text-lg">
Loop: <b>propose → measure → remember → refine</b>
</div>

<!--
SCRIPT:
ECHO has four pieces working in a loop. The Orchestrator is the brain — an LLM that proposes neighbor sets, conditioned on stock metadata, a similarity graph, and memory from past experiments. The Forecast Service is the measuring instrument — it always uses the same calendar-aligned features and the same chronological split, regardless of which backbone is selected. The Experiment Logger guarantees that every proposal and every outcome is auditable. The Analyzer turns the log into dashboards. The cycle is propose, measure, remember, refine.

PROBABLE Q&A:
- Q: "Why an LLM and not a rule-based selector?"
  A: The LLM reasons over heterogeneous evidence — structured metadata, graph priors, memory of past experiment outcomes, and qualitative news sentiment — in a way a rule is hard to extend to. We also enforce structured outputs through schema validation.
- Q: "Which LLM?"
  A: GPT-5.4 for proposals and GPT-5.4-mini for news/sentiment in this manuscript; the planner is provider-pluggable.
-->

---

# Orchestrator — what the LLM actually sees

For each request \((i, H, b)\) the planner gets a typed evidence tuple:

$$\mathcal{C}_{i,H,b} = (m_i,\;\mathcal{U}_i,\;\mathcal{G}_i,\;\mathcal{M}_{i,H},\;s_i)$$

| Symbol | Meaning |
|--------|---------|
| \(m_i\) | Target metadata (sector, mcap bucket, volatility bucket, correlations) |
| \(\mathcal{U}_i\) | De-duplicated candidate pool (sector ∪ mcap ∪ top-corr) |
| \(\mathcal{G}_i\) | Top-5 graph neighbors by learned edge weight |
| \(\mathcal{M}_{i,H}\) | Best historical configurations for the same target & horizon |
| \(s_i\) | News-sentiment object (label + summary + mood index in [0, 1]) |

Search runs in **3 stages**: **exploration → refinement → mutation**, each emitting structured proposals scored by `mae_for_ranking`.

<!--
SCRIPT:
The planner doesn't free-form guess. It receives a structured evidence tuple: target metadata, candidate pool, graph priors, memory of past best runs, and a sentiment object derived from recent headlines. Search runs in three stages — exploration, refinement, mutation — and each stage's best result is fed back as memory into the next. Proposals are validated against a JSON schema before execution.

PROBABLE Q&A:
- Q: "Could the LLM hallucinate symbols?"
  A: All proposals go through schema validation; unknown symbols are skipped by symbol_exists() in the forecast service before training.
- Q: "What stops the LLM from gaming MAE?"
  A: The ranking score adds a penalty (lambda = 1.0) when MAE fails to beat the zero-return baseline, so trivial near-zero predictions cannot dominate selection.
-->

---

# Similarity graph — the structural prior

An undirected weighted graph \(G = (V, E)\) with weight

$$w_{ij} = 0.4\,I^{\text{corr}}_{ij}\rho_{ij} + 0.2\,I^{\text{sec}}_{ij} + 0.2\,I^{\text{vol}}_{ij} + 0.2\,I^{\text{cap}}_{ij}$$

- \(I^{\text{corr}}_{ij}\): is j in i's top-5 correlated peers?
- \(I^{\text{sec}}_{ij}\): same sector?
- \(I^{\text{vol}}_{ij}\): same volatility bucket?
- \(I^{\text{cap}}_{ij}\): same market-cap bucket?
- Max possible weight = 1.0; edge added only when \(w_{ij} > 0\).

This graph is **not** the model. It is a **prior** the LLM consults when proposing neighbors.

<!--
SCRIPT:
The graph is a structural prior — it blends raw correlation with categorical agreement on sector, volatility bucket, and market-cap bucket. Correlation alone is brittle; categorical buckets stabilise it. The graph is consumed by the orchestrator, not used directly as a model — that distinction matters.

PROBABLE Q&A:
- Q: "Why these weights — 0.4, 0.2, 0.2, 0.2?"
  A: Correlation gets the highest single weight because it directly measures co-movement; the three categorical terms each get 0.2 so no single descriptor dominates over correlation.
- Q: "Is the graph dynamic?"
  A: The buckets are computed cross-sectionally over the universe at build time; rolling-window correlations would make a fully dynamic graph — future work.
-->

---

# Forecast service — features

For target \(i\) and each neighbor \(j \in \mathcal{N}_i\):

**Target block** (autoregressive):
$$x^{\text{tgt-lag}}_{t,\ell} = r^{(i)}_{t-\ell},\;\ell\in\{1,\ldots,5\}\quad x^{\text{tgt-mean10}}_t,\; x^{\text{tgt-std10}}_t$$

**Neighbor block** (spillover):
$$x^{j,\text{lag1}}_t = r^{(j)}_{t-1}, \qquad x^{j,\text{roll5}}_t = \tfrac{1}{5}\sum_{u=1}^{5}r^{(j)}_{t-u}$$

**Label** — H-day cumulative forward return:
$$y_t = \frac{P^{(i)}_{t+H}}{P^{(i)}_t} - 1$$

Per-row width: \(p = 5 + 2 + 2|\mathcal{N}_i|\). For 2 neighbors → **11** features.

<!--
SCRIPT:
Every experiment uses the same feature recipe: 5 target lags, a 10-day rolling mean and standard deviation on the target, and for each neighbor, exactly two features — a 1-day lag and a 5-day rolling mean. Labels are H-day cumulative forward returns. Neighbor returns are reindexed onto the target's date index before features are computed, which is how we prevent silent calendar drift between symbols.

PROBABLE Q&A:
- Q: "Why only lag-1 for neighbors instead of multiple lags?"
  A: The spillover hypothesis is that information from a neighbor arrives at t-1 and is digested by the target around t. Higher lags add columns without clear additional signal, and they cost statistical power.
- Q: "Why H-day cumulative return and not log return?"
  A: Arithmetic cumulative return is interpretable as percentage P&L; we report MAE in the same units (% of cumulative H-day return).
-->

---

# Leakage-safe protocol — why this matters

<div class="grid grid-cols-2 gap-4">

<div>

### Chronological 80/20 with horizon purge

For \(n = |X|\) and split index \(s = \lfloor 0.8\,n \rfloor\):

$$\mathcal{T}_{\text{train}} = \{1, \ldots, s - H\}$$
$$\mathcal{T}_{\text{test}} = \{s, \ldots, n\}$$

Training ends **H rows earlier** so that train labels (which depend on prices up to \(t+H\)) never overlap the test window.

</div>

<div>

### Naive baselines reported every run

- **Zero-return** \(\text{MAE}_0\): predict 0 always.
- **Train-mean** \(\text{MAE}_\mu\): predict \(\bar{y}_{\text{train}}\).

### Ranking penalty

$$\text{MAE}_{\text{rank}} = \begin{cases}\text{MAE} & \text{if } \text{MAE} < \text{MAE}_0 \\ \text{MAE} + \lambda & \text{else}\end{cases}\;\lambda = 1.0$$

</div>

</div>

<!--
SCRIPT:
This slide is the heart of the methodology. Two things stop us from fooling ourselves. First, the H-row purge gap: because the label at time t uses the price at t+H, the last H rows of training would otherwise leak information into the test set. We just delete that overlap. Second, every single run is scored not in isolation but against the zero-return baseline and the train-mean baseline. And if a model fails to beat zero, its ranking MAE gets a +1.0 penalty so it cannot accidentally dominate selection by being trivially calibrated.

PROBABLE Q&A:
- Q: "Why a +1.0 penalty specifically?"
  A: It is large relative to realistic MAE values (3–8%) so any model failing zero is decisively demoted, while still being a soft penalty rather than disqualification.
- Q: "Is 80/20 a walk-forward split?"
  A: It is a single chronological holdout. We also run expanding-window TimeSeriesSplit CV with two folds and the same horizon-purge gap for stability diagnostics.
- Q: "Why not full walk-forward?"
  A: Compute cost. Acknowledged as future work.
-->

---

# MAE vs MSE — the metrics question (often asked)

<div class="grid grid-cols-2 gap-6">

<div>

### Training loss = MSE
$$\text{MSE} = \tfrac{1}{N}\sum (y_i - \hat{y}_i)^2$$

- XGB default squared-error objective.
- LSTM uses `nn.MSELoss()`.
- **Penalizes large misses heavily.**

</div>

<div>

### Evaluation metric = MAE
$$\text{MAE} = \tfrac{1}{N}\sum |y_i - \hat{y}_i|$$

- Average miss in percentage points.
- Interpretable: "model is off by 4.4% on average."
- **Prevents +/- error cancellation.**

</div>

</div>

<v-click>

**Example** — actual +5%:

| Pred | Signed err | \|err\| | err² |
|------|------------|---------|------|
| +2%  | +3% | 3% | 9 |
| +8%  | −3% | 3% | 9 |
| Signed avg | **0% (misleading)** | | |
| MAE / MSE | | **3%** | 9 |

</v-click>

<!--
SCRIPT:
Two metrics, two roles. We train with MSE because squaring penalizes large misses and gives smoother gradients. We report MAE because it has interpretable units — "the model misses by 4.4% on average for a 10-day cumulative return." The absolute value is critical: without it, a +5% over-prediction and a −5% under-prediction would average to 0%, falsely suggesting perfect accuracy. MAE prevents that cancellation across the entire test set.

PROBABLE Q&A:
- Q: "Why train MSE but report MAE — isn't that inconsistent?"
  A: It's standard in regression. MSE has stable gradients; MAE is the human-readable diagnostic. The model's signed predictions still preserve direction in predicted_return.
- Q: "Does MAE measure direction accuracy?"
  A: Not directly. A wrong-sign prediction usually still creates a large absolute error, but for near-zero returns MAE under-penalizes sign flips. Directional accuracy would be a complementary metric — flagged as future work.
- Q: "Could you train with MAE directly?"
  A: Yes (L1 loss), but it has discontinuous gradient at zero and converges slower in our setup. MSE-train + MAE-report is the practical default.
-->

---

# Dual-backbone evaluation — same data, different model

<div class="grid grid-cols-2 gap-6 mt-4">

<div>

### Boosted trees (XGBoost)
- Consumes \(X \in \mathbb{R}^{n \times p}\) as a flat tabular block.
- 120 trees, max depth 4, lr 0.05, seed 42.
- Default squared-error objective.
- Each row = one independent observation.

</div>

<div>

### Stateless LSTM
- Same features, reshaped into sliding windows of length \(L = 5\).
- 2 layers, 64 hidden units, dropout 0.2, Adam @ \(10^{-3}\).
- Early stopping on chronological validation tail.
- "Stateless" = hidden state **does not carry across experiment requests**.

</div>

</div>

<v-click>

Same calendar-aligned dataset → same labels → same horizon-purged split → **any MAE difference is attributable to model class, not to data handling.**

</v-click>

<!--
SCRIPT:
We deliberately picked two complementary backbones. XGBoost sees each day as a row in a tabular block — it is great at exploiting individual coordinates. The LSTM sees a five-day rolling window of the same features — it can mix information across the local time context. Critically, both consume the same calendar-aligned features and the same horizon-purged split, so any difference in MAE comes from model class, not from preprocessing.

PROBABLE Q&A:
- Q: "What does 'stateless' mean?"
  A: Recurrence happens WITHIN a 5-day window. We do NOT carry the LSTM hidden state from one experiment request to the next, because each request can have a different neighbor set — which means different feature semantics — and continuing the state would be meaningless.
- Q: "Why not a fancier sequence model like Transformer?"
  A: Comparability with XGB. We wanted a fair head-to-head; richer sequence backbones are future work.
-->

---

# Data — what's in the corpus

| Item | Value |
|------|-------|
| Universe | NIFTY 50 constituents |
| Window | 2019 – 2024 |
| Targets covered | **56** unique tickers |
| Sectors covered | **10** |
| Unique predictors used | **56** |
| Horizons evaluated | **H = 10** and **H = 20** trading days |
| Calibrated runs at H=10 | **1,175** |
| Calibrated runs at H=20 | **1,536** |
| Backbones | XGBoost (511 runs at H=10) + Stateless LSTM (664 runs at H=10) |

<!--
SCRIPT:
The corpus is real NIFTY 50 daily close data from 2019 to 2024, with sector and market-cap metadata from a public provider. We cover 56 distinct ticker symbols as both targets and predictors, ten sectors. At H=10 we have 1,175 calibrated experiments; at H=20, 1,536. Both backbones are evaluated on every target.

PROBABLE Q&A:
- Q: "Why 2019–2024?"
  A: Covers pandemic shock, recovery, inflation regime, rate-hike cycle — enough macro diversity to test cross-regime robustness.
- Q: "Why 56 not 50?"
  A: 56 because of historical index reconstitutions and additions we tracked over the window.
-->

---

# Headline results — H = 10

<div class="grid grid-cols-2 gap-6">

<div>

### Dashboard summary
| Metric | Value |
|--------|-------|
| Calibrated experiments | 1,175 |
| Global avg MAE | **4.39%** |
| Best single MAE | 2.43% (Healthcare) |
| Worst single MAE | 16.09% (Cons. Cyclical) |
| Beat-zero (all runs) | 15.15% |
| Beat-zero (best per target) | **48.21%** |

</div>

<div>

### Baseline A/B/Proposed wins (56 targets)
| Backbone | A (target-only) | B (corr.) | **Proposed** |
|----------|----|----|----|
| **XGB** | 12 (21.4%) | 2 (3.6%) | **42 (75.0%)** |
| **LSTM** | 24 (42.9%) | 7 (12.5%) | **25 (44.6%)** |

</div>

</div>

<v-click>

**Read this line out loud**: under XGB, the agent dominates fixed-correlation neighbors **42 to 2**. Under LSTM, agent and target-only are essentially **tied 25 vs 24**.

</v-click>

<!--
SCRIPT:
Let me read these numbers slowly because they carry the main story. Across 1,175 experiments, average MAE is 4.39%. The single best run achieved 2.43% MAE on a Healthcare target. Now look at the per-target competition. Under XGBoost, agent-proposed neighbors win on 42 of 56 targets — that's 75%. Fixed-correlation neighbors win on only 2. But under LSTM, the agent and target-only-features are nearly tied, 25 versus 24. So neighbor enrichment is not free signal — it depends on the model class. The fixed-correlation arm loses under BOTH backbones, which proves that WHICH neighbors are chosen matters, not just whether neighbors are added.

PROBABLE Q&A:
- Q: "Why does the LLM tie target-only under LSTM?"
  A: LSTM already pools temporal information through its hidden state, so additional neighbor columns add less marginal signal than they do for trees, which need explicit informative coordinates.
- Q: "What's 'beat-zero best-per-target'?"
  A: For each target, take its lowest-MAE configuration; check whether THAT MAE beats predicting 0% always. 48.21% of targets clear that bar at H=10.
-->

---

# Headline results — H = 20

<div class="grid grid-cols-2 gap-6">

<div>

### Dashboard summary
| Metric | Value |
|--------|-------|
| Calibrated experiments | 1,536 |
| Global avg MAE | **5.92%** |
| Best single MAE | 2.79% |
| Worst single MAE | 34.45% |
| Beat-zero (all runs) | 24.41% |
| Beat-zero (best per target) | **55.36%** |

</div>

<div>

### Baseline A/B/Proposed wins
| Backbone | A | B | **Proposed** |
|----------|----|----|----|
| **XGB** | 8 (14.3%) | 11 (19.6%) | **37 (66.1%)** |
| **LSTM** | 22 (39.3%) | 6 (10.7%) | **28 (50.0%)** |

</div>

</div>

<v-click>

**Distribution-level mean MAE**
| Backbone | A | B | **Proposed** |
|----------|----|----|----|
| **XGB** | 5.90% | 6.69% | **5.59%** |
| **LSTM** | 5.31% | 6.79% | **5.10%** |

</v-click>

<!--
SCRIPT:
At H=20 the picture sharpens. Average MAE rises to 5.92% — longer horizons are intrinsically harder. But the beat-zero bar also rises with horizon, so 55% of targets clear it at best — that's actually better than at H=10. The Proposed arm again dominates XGB (66%) and now also wins LSTM cleanly (50%). At the distribution level, Proposed has the lowest average MAE under both backbones, and the most targets beating zero — 24 for XGB, 34 for LSTM.

PROBABLE Q&A:
- Q: "Why is beat-zero higher at longer horizon?"
  A: Zero-return MAE roughly scales as sqrt(H) under random-walk assumptions. As MAE_0 grows, the bar to beat zero gets easier — that's a horizon effect, not a model effect.
- Q: "Did model quality actually improve at H=20?"
  A: Average MAE is worse in absolute terms (5.92 vs 4.39), but the SIGNAL relative to zero baseline is stronger. Use relative-to-baseline as the honest measure.
-->

---

# XGB vs LSTM — head-to-head per target

<div class="grid grid-cols-2 gap-6">

<div>

### H = 10
| | XGB | LSTM |
|--|-----|------|
| Avg MAE | 0.0437 | **0.0399** |
| Best MAE | 0.0267 | **0.0243** |
| **Wins (of 56)** | 21 | **35** |

</div>

<div>

### H = 20
| | XGB | LSTM |
|--|-----|------|
| Avg MAE | 0.0601 | **0.0586** |
| Best MAE | 0.0316 | **0.0279** |
| **Wins (of 56)** | 11 | **45** |

</div>

</div>

<v-click>

LSTM wins more **per-target head-to-heads** and has lower average MAE under identical splits.

But fixed-neighbor selection still **loses under both backbones** — so neighbor design is doing real work.

</v-click>

<!--
SCRIPT:
When we hold the neighbor set fixed (always best logged agent set) and vary only the backbone, the stateless LSTM wins more per-target battles at both horizons — 35/56 at H=10 and 45/56 at H=20 — with lower average AND best MAE. This is one of the key empirical surprises: the LSTM was supposed to be the "weaker" arm because the same agent neighbors didn't help it as much in A/B/Proposed, but in raw model-vs-model, the LSTM extracts more signal from the same features. So XGB benefits more from neighbor enrichment, but LSTM extracts more juice per neighbor set on average.

PROBABLE Q&A:
- Q: "So which model should I use?"
  A: If the goal is "given best neighbors, smallest MAE" — LSTM. If the goal is "neighbor design matters and I want clear separation between A/B/Proposed" — XGB shows the contrast more dramatically. Both are honest.
- Q: "Is this a contradiction with the previous slide?"
  A: No. A/B/Proposed measures HOW MUCH neighbors help WITHIN a backbone. Head-to-head measures BACKBONE QUALITY with neighbors held fixed. They answer different questions.
-->

---

# Sector influence — where signal actually lives

<div class="grid grid-cols-2 gap-6">

<div>

### Strongest cross-sector at H=20
(min 10 experiments per cell)

| Predictor → Target | Score | n |
|---|---|---|
| Healthcare → Cons. Defensive | **+35.83** | 50 |
| Technology → Financial Svc | +31.76 | 33 |
| Energy → Communication Svc | +31.73 | 18 |
| Industrials → Comm. Svc (H=10) | +37.6 | — |

</div>

<div>

### Weakest / harmful cells at H=20

| Predictor → Target | Score | n |
|---|---|---|
| Consumer Cyclical → Energy | **−80.49** | 25 |
| Basic Materials → Cons. Cyclical | −41.83 | 74 |
| Consumer Cyclical → self | −35.49 | 522 |

</div>

</div>

<v-click>

**Score = % improvement vs global average MAE for that sector pair.**

→ Consumer Cyclical self-prediction is **the single worst cell** — they reach **across** sectors, not within.

</v-click>

<!--
SCRIPT:
This is the dashboard's most informative visualization for non-specialists. We aggregate MAE by (predictor sector, target sector) pair and report it as percentage improvement over the global average MAE. Warm cells are predictive sector pairings; cool cells are harmful. Two patterns to highlight. First, cross-sector positive cells like Healthcare → Consumer Defensive at +35.83% are stronger than most within-sector relationships, which means relevant signal is NOT where you'd guess. Second, consumer cyclical stocks predict themselves WORSE than the global average — by 35%. That's why the agent reaches outside sector for cyclicals, and that's exactly why fixed-correlation (which usually picks same-sector) loses on cyclicals.

PROBABLE Q&A:
- Q: "Why is same-sector sometimes worse?"
  A: Same-sector firms move together at the macro level but their idiosyncratic news cancels signal at the lag-1 level. Cross-sector lead-lag (e.g., commodities lead manufacturers) survives the lag better.
- Q: "What's a score of +35 actually mean?"
  A: Experiments with this sector pair achieved average MAE 35% lower than the global average MAE.
-->

---

# Sector influence heatmap — H=10

<div class="flex justify-center">
<img src="/predictor_to_target_sector.png" class="max-h-100" />
</div>

<!--
SCRIPT:
Visual summary. Warmer cells indicate that experiments using a predictor from that ROW sector to forecast a target in that COLUMN sector achieved MAE materially below the global average. Cooler cells show the opposite. The strongest single cell at H=10 is Industrials → Communication Services at +37.6%. The worst is Consumer Cyclical self-prediction at −49.6%. The visual diagonal is NOT uniformly warm — that is the key insight that motivates cross-sector neighbor search.

PROBABLE Q&A:
- Q: "Is this just noise on small cells?"
  A: Empty cells are filtered out for under-supported pairs. Reported numbers are for cells with enough experiments to summarise.
-->

---

# Case study — Success: ICICIBANK

<div class="grid grid-cols-2 gap-6">

<div>

### Agent-proposed neighbors
\{HINDUNILVR, INDUSINDBK, SBILIFE\}

- Mix of **Consumer Defensive**, **Financial Services**, **Insurance**.
- Cross-sector, not within-sector — the LLM picked **macro-financial liquidity proxies**.

</div>

<div>

### Result at H=10

| Arm | MAE |
|-----|-----|
| Target-only (A) | 0.0273 |
| Fixed-corr (B) | 0.0596 |
| **Proposed** | **0.0267** |
| Zero baseline | beaten ✓ |

MAE std across experiments: **0.0008** → very stable signal.

</div>

</div>

<!--
SCRIPT:
ICICIBANK at H=10 — agent proposed Hindustan Unilever, IndusInd Bank, and SBI Life. That mix is interesting: it spans Consumer Defensive, Financial Services, and Insurance. The agent's rationale was that these are macro-financial liquidity proxies. Result: MAE 2.67%, beating target-only (2.73), fixed-correlation (5.96), and the zero baseline. Crucially the standard deviation across multiple agent proposals for this target was only 0.0008 — meaning the agent found a stable signal, not a lucky outlier.

PROBABLE Q&A:
- Q: "Why does HINDUNILVR help ICICIBANK?"
  A: Hindustan Unilever is a defensive consumption bellwether that reacts to liquidity and rate cycles, which feed into bank earnings expectations. The signal is plausible economically, but our claim is statistical fit on the holdout, not causal.
-->

---

# Case study — Challenge: TRENT

<div class="grid grid-cols-2 gap-6">

<div>

### Why this is hard

- Sector: **Consumer Cyclical** (worst sector in matrix).
- Driven by sentiment + idiosyncratic news.
- Lagged-return basis cannot fully capture story arcs.

</div>

<div>

### Result at H=10

| Arm | MAE |
|-----|-----|
| Target-only (A) | 0.0628 |
| Fixed-corr (B) | 0.1283 |
| **Proposed** | **0.0625** ✓ |

MAE still **6.25%** (≈ 6.3% cumulative 10-day return error).

MAE std across experiments: **0.039** → high dispersion, regime-dependent.

At H=20 → Proposed = 15.0%, A wins at 10.2%.

</div>

</div>

<!--
SCRIPT:
TRENT is the honest counter-example. Even though the agent wins versus both baselines at H=10 — 6.25% MAE versus 6.28% and 12.83% — the absolute error remains high. The standard deviation across experiments is 0.039, fifty times higher than ICICIBANK. That means the agent finds neighbors that work in some regimes and fail in others. This is consistent with the sector matrix showing Consumer Cyclical self-prediction at −35.49% at H=20. The honest conclusion is: ECHO improves what is improvable, but cannot rescue stocks whose dynamics are driven by news flow outside the lagged-return basis.

PROBABLE Q&A:
- Q: "Why don't you just add news features?"
  A: We do include news sentiment in the orchestrator's planning context, but the model itself trains on returns only. Adding NLP-derived features to the trainable feature set is in future work.
- Q: "Why does H=20 break TRENT?"
  A: Longer horizons compound idiosyncratic shocks; high-beta cyclicals also accumulate more directional drift, widening the achievable MAE band.
-->

---

# Failure modes — what does NOT work

<v-clicks>

1. **High-volatility cyclicals** — TRENT, TMPV, TMCV. Best achievable MAE 6–8%, 2–3× worse than stable large caps.
2. **Regime shifts** — CV worst-split analysis flags 2020-Q1/Q2 (pandemic) and 2022-H2 (rate-hike) as systematic stress periods.
3. **Sector isolation** — Consumer Cyclical self-prediction = −49.6 (H=10), −35.49 (H=20). Within-sector heuristics actively hurt.
4. **Backbone-specific neighbors** — A great XGB neighbor set is not automatically a great LSTM neighbor set.
5. **Shared hard targets** — TRENT, TMPV, TMCV are bad under **both** backbones — the binding constraint is the feature basis, not the model.

</v-clicks>

<!--
SCRIPT:
We do not hide failures. Five categories. One — high-volatility consumer cyclicals are intrinsically harder. Two — regime shifts hurt across the board; we flag this through CV worst-split diagnostics. Three — within-sector heuristics actively harm cyclicals; you can lose by being "obvious." Four — the same neighbor set does not transfer perfectly across backbones. And five — some targets are hard regardless of model, which tells us the FEATURES need extending, not just the model.

PROBABLE Q&A:
- Q: "If models fail in regime shifts, isn't the whole pipeline fragile?"
  A: That's exactly why we report CV worst-split MAE and date intervals per experiment — to expose regime fragility instead of averaging it away. It is a known cost; the solution is regime-aware features.
- Q: "What's the worst-case MAE you observed?"
  A: 16.09% at H=10, 34.45% at H=20. Both are flagged as outliers in the logs.
-->

---

# What is novel about ECHO

<v-clicks>

- **Integration**, not a single new model: orchestration + structured context + experiment logging + identical evaluation contract across two backbones.
- **Evaluation hygiene as a first-class artifact**: calendar alignment, horizon purge, zero-baseline penalty, full provenance logging.
- **Backbone-dependent finding**: same neighbors help XGB strongly and LSTM modestly — quantified, not just observed.
- **Auditable agent**: every proposal carries `experiment_id`, rationale, source, round, backbone — full reproducibility of the search trajectory.

</v-clicks>

<!--
SCRIPT:
We don't claim a brand-new forecasting algorithm. The contribution is integration and discipline: orchestration plus structured evidence plus identical evaluation across backbones plus full logging. And we demonstrate that backbone-dependence is REAL — not vague, but quantified across 1,175 + 1,536 experiments. The auditability piece matters because every agent proposal can be traced end-to-end through the JSON log.

PROBABLE Q&A:
- Q: "What's the academic novelty?"
  A: Three parts — leakage-safe shared-evaluation contract across backbones, calibrated baseline-aware ranking penalty, and empirical demonstration of backbone-dependent neighbor value.
- Q: "Is this just packaging existing tools?"
  A: The components individually are known; the combination — agent + leakage-safe evaluator + memory loop with full audit trail — has not been demonstrated on Indian equities at this scale.
-->

---

# Limitations — said up front

<v-clicks>

- **Feature basis is lag-only.** Fundamentals, order-flow, news features could lift the ceiling.
- **Two horizons only.** H = 10 and H = 20 — generalising the curve to H ∈ {5, 30, 60} is left for future work.
- **No walk-forward over years.** Single 80/20 + 2-fold CV; full rolling-window walk-forward is on the roadmap.
- **No transaction-cost / portfolio evaluation.** MAE is a research diagnostic, not P&L.
- **LLM dependence.** Performance can drift with planner model updates; we mitigate via schema validation but do not eliminate.
- **Indian market only.** Cross-market generalisation untested.

</v-clicks>

<!--
SCRIPT:
Six limitations, stated proactively. The feature set is intentionally lean — lagged returns plus rolling statistics — and that caps the ceiling. Two horizons only, single chronological holdout plus two-fold CV, no walk-forward yet, no transaction-cost modelling, dependency on a specific LLM version, single market. These are honest gaps and they map directly onto the next-steps slide.

PROBABLE Q&A:
- Q: "If you have no walk-forward, how do you know it's not just luck?"
  A: We have TWO orthogonal stability signals — the 2-fold CV with worst-split diagnostics, and the per-target win counts across 56 targets. Both would need to be wrong simultaneously for the finding to be luck.
- Q: "Is this a replacement for backtesting?"
  A: Explicitly no. MAE on returns is a research diagnostic, not a trading proof. The next step is portfolio-level walk-forward backtest with costs.
-->

---

# Future work

<v-clicks>

1. **Walk-forward evaluation** with rolling-window retraining.
2. **Richer features**: order-flow proxies, intraday volatility, news embeddings, fundamentals.
3. **Portfolio-level metrics**: turnover, transaction costs, capacity.
4. **Dynamic similarity graph** with rolling-window correlations.
5. **Backbone expansion**: lightweight Transformer / TFT / N-BEATS comparison.
6. **Directional metric**: hit rate + signed regression in parallel with MAE.

</v-clicks>

<!--
SCRIPT:
Six clear next steps. Walk-forward retraining for full temporal robustness. Richer features beyond lagged returns. Portfolio-level evaluation. Dynamic graph instead of static one. More backbones for comparison. And a directional metric alongside MAE to address the "MAE is not the whole story" critique. Each one is a clean follow-up project.

PROBABLE Q&A:
- Q: "Which one would you tackle first?"
  A: Walk-forward + directional metric. Both are cheap to implement and remove the two most common reviewer objections.
-->

---

# Live dashboard

<div class="flex justify-center">
<img src="/dashboard_full.png" class="max-h-100" />
</div>

<div class="text-sm opacity-70 mt-2">
Calibrated leaderboard · top predictors · sector heatmap · signal network · stock explorer with prediction overlays.
</div>

<!--
SCRIPT:
This is the analyzer dashboard — calibrated leaderboard per backbone, predictor frequency, sector matrix, force-directed signal graph, and an interactive stock explorer. Everything you've seen so far is generated from the same append-only experiment log, so the dashboard is always a faithful view of what actually ran.

PROBABLE Q&A:
- Q: "Can examiners try the dashboard?"
  A: Yes — runs locally, served from the analyzer output. Code and logs are in the repo.
-->

---

# Conclusions

<v-clicks>

- **What we built**: ECHO — agentic experimentation loop for cross-stock neighbor discovery on NIFTY.
- **What we measured**: 1,175 + 1,536 calibrated experiments across two horizons under two backbones with identical leakage-safe protocol.
- **What we found**:
  - Agent-proposed neighbors **dominate fixed correlation** in **all four** (backbone × horizon) settings.
  - Their **lift over target-only** is **backbone-dependent** — strong for XGB, modest for LSTM.
  - **LSTM** has lower **typical and best** MAE in head-to-head; **XGB** shows the **largest neighbor-design contrast**.
  - **Fixed correlation is consistently the weakest arm** — which neighbors you pick matters.

</v-clicks>

<v-click>

> **Bottom line**: ECHO turns neighbor discovery into an **auditable experimental-science workflow**, with measured wins, measured losses, and full provenance.

</v-click>

<!--
SCRIPT (closing — 30–45 sec):
To close: ECHO is a reproducible agentic experimentation loop. Across more than 2,700 calibrated experiments at two horizons, we have four robust findings — agent neighbors beat fixed correlation everywhere, the lift over target-only is backbone-dependent, LSTM extracts more juice from the same features, and naive correlation neighbors actively hurt. The deeper contribution is the methodological discipline that lets us say all of that honestly: shared splits, baselines, ranking penalty, and full audit trail. Thank you — happy to take questions.

PROBABLE FINAL Q&A:
- Q: "If you had one more month, what would you do?"
  A: Walk-forward + transaction-cost portfolio backtest. That converts MAE diagnostics into economic claims.
- Q: "What was the hardest engineering problem?"
  A: Calendar alignment — Indian trading calendar with corporate-action gaps. We index by trading date and inner-join, never by row position, which was a deliberate scientific guardrail.
- Q: "What was the hardest research decision?"
  A: Picking MAE as the headline metric. We considered MSE/RMSE but MAE is interpretable in % of cumulative return — a reviewer-friendly unit.
-->

---
layout: center
class: text-center
---

# Thank you

### Questions?

<div class="text-sm opacity-70 mt-6">
ECHO — LLM-Orchestrated Neighbor-Enriched Forecasting<br>
Repository: <code>BMondal-dev/financial-agent</code>
</div>

<!--
HAND-OFF Q&A CHEATSHEET (for the presenter):

1. "Why MAE not RMSE?"
   → MAE is in % of cumulative return — reviewer-friendly. RMSE squares errors and inflates outlier influence.

2. "Why MSE for training but MAE for evaluation?"
   → MSE has stable gradients and convex curvature; MAE has discontinuous gradient at zero. Standard practice in regression.

3. "Why ≤ 3 neighbors?"
   → Per-row width grows as 5 + 2 + 2k; small k keeps the design matrix tractable and proposals comparable.

4. "Why XGBoost AND LSTM?"
   → Different inductive biases — trees exploit coordinates, LSTMs pool temporal context. Backbone-dependence of neighbor value is itself a finding.

5. "What is the ranking penalty for?"
   → Stops models that fail to beat the zero-return baseline from dominating selection through trivially-small predictions.

6. "How is leakage prevented?"
   → Chronological split + H-row purge gap so train labels (which depend on prices up to t+H) never overlap the test window.

7. "Why no walk-forward?"
   → Compute cost. Single 80/20 + 2-fold time-series CV with horizon purge gives stability signal at acceptable cost. Walk-forward is in future work.

8. "How does the LLM not hallucinate symbols?"
   → JSON schema validation + symbol_exists() guard in forecast service; unknown symbols are dropped before training.

9. "Why static similarity graph?"
   → Categorical buckets stabilise the prior. Rolling-window graph = future work.

10. "Why 'stateless' LSTM?"
    → Recurrence WITHIN 5-day windows. NO hidden state propagated across different experiment requests because feature semantics change with neighbor set.

11. "What's the single most important number?"
    → 42/56 agent-vs-fixed-correlation wins under XGB at H=10. That's the clearest signal that neighbor SELECTION (not just inclusion) matters.

12. "What did you find by accident?"
    → That LSTM extracts more juice from the same features (45/56 wins at H=20) while XGB shows the largest A/B/Proposed contrast. Two true things in tension.

13. "How long did data collection take?"
    → Daily close + metadata, automated pipeline; the bottleneck was schema discipline (calendar alignment) not raw ingest.

14. "What's the dashboard for?"
    → Same log powers both the agent's memory loop and the human analyst's report — single source of truth, no separate reporting pipeline.

15. "Production-ready?"
    → As a research instrument, yes. As a trading system, no — needs PnL/cost layer and walk-forward.
-->
