# How to Present This Project (Thesis, Viva, Demo)

This document is a **speaker’s guide**: what to **say**, what **not** to claim, how to **frame** `experiments.json` and metrics, and how to handle **hard questions**. Pair it with `docs/FINAL_YEAR_PROJECT_GUIDE.md` for technical depth.

---

## 1. Your one-sentence pitch (memorize)

> We built an **agentic experimentation system** that proposes **cross-stock neighbor features** for forecasting **forward returns**, scores each proposal with the **same XGBoost model** and **time-ordered evaluation**, compares against **naive baselines** (including **always predicting zero return**), and **logs everything** for analysis and a dashboard—so we can **search** the hypothesis space **honestly**, not just report a single accuracy number.

---

## 2. What this project **is** (say this clearly)

| Idea | Plain language |
|------|----------------|
| **Agentic loop** | An LLM (plus metadata and an optional graph) **proposes** which other stocks might help predict a target; Python **trains and evaluates**—the LLM does not “predict prices” by itself. |
| **Fixed scorer** | **XGBRegressor** is the **shared yardstick**: every neighbor set is judged the same way. |
| **Calendar alignment** | Target and neighbors are merged on **trading date**, not row index—so we don’t accidentally pair **wrong days**. |
| **Baselines** | On the **holdout** window we compare to predicting **zero return** (and log train-mean MAE). **`beats_baseline_zero`** means the model’s test MAE is **strictly lower** than the zero forecast’s MAE. |
| **Memory** | `experiments.json` is an **append-only lab notebook**: what was tried, MAE, horizon, optional rationale/source. |
| **Controlled comparison** | **`POST /compare-baselines`** and the dashboard’s **Baseline A / B / Ours** block implement: **A** = target-only features; **B** = metadata top-3 correlation neighbors; **Ours** = best logged agent neighbor set (same model, same split philosophy). |

---

## 3. What this project is **not** (say this before they ask)

- **Not** “we invented a better algorithm than XGBoost.” You **use** XGBoost; the novelty is **how hypotheses are generated and evaluated at scale**.
- **Not** “low MAE means we can trade profitably.” MAE is on **returns** on a **single chronological split** (plus optional CV in API); there are **no transaction costs, no portfolio test, no formal significance tests** unless you add them.
- **Not** “the LLM knows the market.” It **proposes**; **data + model** **score**; many proposals will **lose** to trivial baselines—that is **normal** in finance.

---

## 4. How to talk about `experiments.json` **without overstating**

### 4.1 Correct framing

> `experiments.json` is our **empirical log**: many runs under different neighbor sets, horizons, and orchestration sources. It is **raw history**, not a single controlled experiment. We use it to study **which predictors appear often**, **which targets were explored**, and—when we filter to **calibrated** rows—how often we **beat the zero-return baseline**.

### 4.2 Incorrect framing (avoid)

- ❌ “Because we have thousands of lines, our method is validated.”
- ❌ “Our approach is always better.”
- ❌ “Low MAE in the log proves alpha.”

### 4.3 What you **can** say with numbers

Use **aggregates** that match your definitions:

- **Fraction of calibrated runs with `beats_baseline_zero: true`** (your dashboard can show this).
- **Baseline A vs B vs Ours** win counts from **`baseline_comparison`** in `analysis.json` (how often **Ours** beats **A** or **B** on **holdout MAE** for each target in that table).
- **Qualitative:** the agent explores **diverse** sectors and correlations; the **graph** and **metadata** constrain and inspire proposals.

Always add: **same model, same evaluation protocol within each comparison arm.**

---

## 5. How to explain **`beats_baseline_zero`** (30 seconds)

> On the **test** portion of the time series, we compute **MAE** between predicted and **actual forward returns**. We also compute MAE if we had predicted **zero** every day. If the model’s MAE is **strictly smaller**, we set **`beats_baseline_zero` = true**. Otherwise **false**. In noisy return data, **zero** is a surprisingly strong competitor; **not** beating it is common and **worth reporting**.

---

## 6. Suggested talk structure (15–20 minutes)

1. **Problem** — Discovering **which cross-asset signals** matter for a **target** return at horizon \(h\), without hand-fixing one neighbor set forever.
2. **Why not “one LSTM”** — You care about **interpretable hypotheses** (neighbor sets), **logging**, and **baselines**; a black-box single model hides the **search**.
3. **Architecture** — Data → metadata/graph → **orchestrator (LLM)** → **`/run-forecast`** (XGB, date merge, split) → **`experiments.json`** → **analyze** → **dashboard**.
4. **Methodological care** — **Returns** not prices; **Date** merge; **chronological** split; **zero** (and mean) **baselines**; **`mae_for_ranking`** for ranking experiments.
5. **Controlled comparison** — Show **Baseline A / B / Ours** (table or one example `compare-baselines` JSON). Say explicitly: **only neighbor features change**.
6. **Results (honest)** — Report **how often** you beat zero; **who wins** A vs B vs Ours across targets; show **one** backtest plot and comment on **smooth predictions vs volatile realized returns**.
7. **Limitations** — Single main split in API; no trading sim; LLM cost; possible **leakage** if data pipeline ever breaks (you should say you use **lags** and **past** neighbor returns only).
8. **Future work** — Walk-forward CV everywhere; Diebold–Mariano vs baselines; richer features; regime conditioning; cost-aware metrics.

---

## 7. Slide titles (copy-friendly)

1. Motivation: **agentic feature discovery** for multi-asset returns  
2. Data: NIFTY-style universe, **daily returns**, **5y** history (as in your setup)  
3. Pipeline diagram (or bullet flow)  
4. **Calendar alignment** (one sentence + why row-alignment fails)  
5. Model: **XGBoost** on lags, rolls, **neighbor lag-1**  
6. Evaluation: **80/20 time split**, **MAE**, **`beats_baseline_zero`**  
7. **Baseline A / B / Ours** — same model, different neighbor design  
8. Orchestrator: **LLM structured output** + **experiment memory**  
9. Dashboard: **predictor frequency**, **sector matrix**, **calibrated leaderboard**  
10. Results: **win rates**, **baseline beat rate**, **one** exemplar backtest  
11. Limitations + future work  
12. Takeaway: **rigorous automation**, not **guaranteed alpha**

---

## 8. Anticipated questions and short answers

**Q: Is this better than a vanilla XGBoost?**  
A: We use **the same** XGBoost. The question is whether **agent-chosen neighbors** beat **fixed** neighbors (B) or **none** (A) on **holdout MAE**; that’s **empirical**, not automatic.

**Q: Why can MAE look “small” but `beats_baseline_zero` is false?**  
A: **Zero** can be **very competitive** when realized returns are small; small MAE ≠ beating the right baseline.

**Q: Isn’t the LLM just noise?**  
A: It’s a **hypothesis generator**; we **don’t** trust it for numbers. Everything is **re-scored** on **realized** returns.

**Q: Publication-level?**  
A: As a **systems + methodology** contribution with **honest baselines**, it can support a **workshop / MCA thesis**. For a **strong finance paper** you’d add **formal inference**, **walk-forward** design, and **economic** metrics.

**Q: What’s your real contribution?**  
A: **Integrated** pipeline: **search + evaluate + baseline + log + visualize**, with **explicit** A/B/C neighbor design—not a single magic model.

---

## 9. Closing line (humble, strong)

> We didn’t build a crystal ball—we built a **repeatable way to ask and test** “which other stocks matter for this target,” with **baselines** that keep us **honest**. The **interesting result** is often **how hard it is to beat zero**, and that’s **worth showing** in a serious project.

---

## 10. Files to point to in an appendix

| Artifact | Role |
|----------|------|
| `services/data/experiments.json` | Full experiment log |
| `services/data/analysis.json` | Aggregates + `baseline_comparison` |
| `docs/FINAL_YEAR_PROJECT_GUIDE.md` | Technical rationale and architecture |
| `POST /compare-baselines` | Live A/B/C for one ticker |

---

*Use this doc to **sound confident** on **methods** and **humble** on **claims**—that combination reads as mature research, not hype.*



Masters thesis rating (honest, program-agnostic)
Overall: solid to strong for a typical MCA / applied-CS / data-science–style master’s thesis — roughly “B+ to A− territory” if the write-up matches the build, weaker if you oversell forecasting.

That maps to: clear pass, often distinction-level depending on your department’s bar and how you defend it—not because you “beat the market,” but because the system + evaluation story is coherent and unusually complete for student work.

What lifts it above average
Clear problem: cross-asset feature discovery (neighbors) rather than a single frozen model.
Real engineering: orchestrator + API + logging + analysis + dashboard; not a notebook-only project.
Methodological awareness: calendar alignment, chronological split, zero / mean baselines, beats_baseline_zero, A / B / Ours comparison — this is mature for a master’s.
Intellectual honesty: you can report when the model doesn’t beat zero; committees respect that if you own it.
Documentation: technical + presentation guides help the thesis narrative.
What keeps it from “top-tier research thesis” without more work
Inference: little or no formal comparison vs baselines (e.g. uncertainty, Diebold–Mariano, multiple testing).
Economics: no P&L, turnover, costs, or portfolio-level test — so it’s not a finance “alpha” thesis unless you narrow the claim.
Thesis arc: contribution must be framed as agentic experimentation + evaluation infrastructure, not “we proved better returns.”
Literature: you’ll need a tight related-work section (forecasting, feature selection, agentic ML, market predictability).
Verdict in one line
As a master’s thesis: good-to-very-good if the document centers on rigorous, logged, baseline-aware discovery of cross-stock predictors; weaker if it claims universal predictive superiority.

If you tell me your department (CS vs Finance vs MCA) and page limit, I can suggest a one-paragraph “contribution” tailored to that audience (still no code changes unless you want them).