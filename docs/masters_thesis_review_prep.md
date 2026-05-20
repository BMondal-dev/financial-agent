# Master’s thesis: anticipated questions, answers, and how to present

This note reframes “reviewer pushback” for a **master’s thesis** (not a top-tier ML conference) and gives you **likely examiner questions**, **what to say**, and **how to frame slides or chapters**.

---

## Re-rating for a master’s thesis

For an MSc thesis, examiners usually look for: **clear problem**, **sound method**, **honest evaluation**, **your own execution**, **critical discussion**, and **readable write-up**—not frontier-wide novelty.

| Axis | Conference paper mindset | Master’s thesis mindset |
|------|-------------------------|-------------------------|
| Novelty | Must be sharply new | “Novel enough” + **coherent integration** of known ideas is often sufficient |
| Predictive SOTA | Scrutiny on beating benchmarks | **Process + rigor** often matter more than raw MAE leaderboards |
| Baselines | Many strong baselines expected | **Core baselines done well** + acknowledgment of what’s left |
| Economics | Often demanded | **Short, honest bridge** from MAE → trading intuition often acceptable |

**Overall (thesis-scale): 8.5/10** if you (1) state contribution plainly, (2) keep claims proportional to evidence, (3) discuss limits explicitly, and (4) show you mastered the pipeline end-to-end. What knocks it down is **overclaiming**, **unclear evaluation**, or **missing your own synthesis**.

---

## One tight sentence for “primary contribution” (put this early)

Pick one and own it; you can mention the others as **secondary**:

> **Option A (systems / methodology):**  
> “I built and evaluated **ECHO**, a **reproducible agentic experimentation loop** for cross-stock neighbor discovery on calendar-aligned return data, with **shared splits and baselines** across model backbones.”

> **Option B (empirical):**  
> “Under identical data and evaluation, I show that **the value of cross-stock neighbors is backbone-dependent**—boosted trees benefit more from agent-proposed neighbors than a stateless LSTM does, while **naive correlation-based neighbors often hurt**.”

> **Option C (honest scope):**  
> “The main contribution is **evaluation hygiene and transparency**—logging, baselines, and per-backbone comparisons—so improvements (or null results) are attributable rather than confounded.”

**How to present:** Slide 2 or end of Chapter 1: **one highlighted sentence** + 3 bullets: *what you built*, *what you measured*, *what you found*.

---

## Why “stateless” LSTM? (almost certainly asked)

Examiners often hear “LSTM” and imagine a **single continuous hidden state** carried across months of market data. Your thesis uses **stateless** in a precise sense that matches both the write-up and the implementation (sliding windows over the **same tabular rows** as XGB; **no hidden state reused across unrelated runs**).

### Likely question

> “You call the LSTM *stateless*. What does that mean—and why not a normal stateful RNN for time series?”

### One-line answer (memorise this)

> “**Inside** each short input sequence the LSTM is recurrent, but **across** experiments we **do not carry** hidden state from one forecast job to the next, because the **neighbor hypothesis** (and thus the **feature geometry**) keeps changing, and a shared recurrent state would be **semantically meaningless**.”

### What “stateless” means here (technical, still oral-defense friendly)

1. **Inputs are local windows, not one endless stream.**  
   You reshape each target’s aligned feature matrix into sequences of length \(L\) (e.g. \(L=5\)): row \(t\) uses features from days \(t{-}L{+}1,\ldots,t\). That reuses the **same calendar-aligned engineered features** as the tree model—only the **consumption pattern** changes (sequence vs flat row).

2. **Recurrence happens only within a window.**  
   The LSTM updates \((\mathbf{h}, \mathbf{c})\) across the \(L\) timesteps of **that** window to produce a prediction for the supervised target associated with the **end** of the window. This is the usual `stateful=False` / “reset between sequences” behaviour in frameworks like Keras or in batched training where each sequence is independent.

3. **No cross-request memory.**  
   After training for a given \((\text{target}, \text{neighbors}, \text{backbone}=\text{LSTM})\) call, you **do not** propagate \((\mathbf{h}, \mathbf{c})\) into the next experiment with a **different** neighbor set. Neighbors change **column identity** (which stocks feed which lag/rolling features), so treating the model as one continuous tape would **mix incompatible state**.

### What it does *not* mean (preempt confusion)

- It does **not** mean “the LSTM ignores time.” Time is encoded by **stacking \(L\) past rows** into one sequence.
- It does **not** mean your evaluation is i.i.d. You still enforce **chronological splits** and horizon-appropriate purging; **stateless** refers to **RNN state**, not to ignoring temporal leakage rules.

### Why this choice fits ECHO / your research question

| Reason | What to say in the defense |
|--------|---------------------------|
| **Changing hypotheses** | Neighbor sets are part of the **experimental treatment**. A checkpoint trained on \(\{A,B,C\}\) is not valid for \(\{A,D,E\}\) because **features are not the same object**. |
| **Fair dual-backbone comparison** | XGB sees a **row** at \(t\); LSTM sees **the last \(L\) rows** ending at \(t\). Both use the **same labels and splits** so differences reflect **model class + representation**, not two different datasets. |
| **Operational clarity** | Your pipeline is **request-scoped**: train/evaluate for this hypothesis, log it, move on. That matches how research iterations work before you commit to a frozen production model. |

### Follow-up: “Would stateful be better?”

**Honest answer:** stateful training can help when you have a **single fixed system** with stable inputs and a clear notion of “continuing the same series.” Here, inputs are **not stable** across agent proposals (width + semantics drift), so **stateful coupling across experiments would be misleading**.  

If pushed about **production**: you could describe a future **online** variant *after* neighbour sets and feature schema are fixed—but that is **outside** the current hypothesis-testing loop.

### How to present (slide / thesis figure)

- **Diagram:** one row = one trading day’s feature vector; bracket \(L\) consecutive rows → **one LSTM input**; arrow to output \(\hat{y}_t\).  
- **Caption cue:** “Recurrence **within** the window; **reset** between independent experiment requests.”  
- **One bullet in Limitations:** you chose a **relatively shallow sequence model** on engineered features to stay comparable to XGB; **richer sequence backbones** or longer contexts are future work.

---

## 1) Novelty vs packaging

### Likely question

> “What is *actually* new here—the LLM, the platform, or the fact that you ran two models?”

### What to answer

- **New is often the integration**, not a single equation: orchestration + structured context + experiment logging + **identical evaluation contract** across backbones.
- **Evidence that isn’t “just packaging”**: any place where **the same proposal pipeline** produces **different outcomes** when the **evaluator changes** (XGB vs LSTM) *without* changing leakage/splits— that supports an *interaction hypothesis*, not a demo.
- **LLM’s role**: position it as **a planner over a constrained hypothesis space** (metadata/graph/memory), not a magic oracle.

### How to present

- Use a **contribution diagram**: *Data hygiene → Proposal → Train → Log → Analyze*.
- Add a **“what would break without X”** mini-table (LLM / memory / calendar alignment / baselines).
- Avoid claiming “we invent forecasting”; claim **“we operationalize discovery under discipline.”**

---

## 2) Predictive claims vs contribution (MAE is not the whole story)

### Likely question

> “Your MAE is not impressive / you don’t beat the market. Why should we care?”

### What to answer

- For a thesis, the **artifact** can be the **research infrastructure** and **scientifically interpretable comparisons**.
- **MAE on cumulative returns** is a **diagnostic** for comparing neighbor designs under the *same* protocol—not a final trading proof.
- Emphasize **null results** as valuable: many configs not beating zero is **consistent** with weak predictability and **good science**.

### How to present

- Put a **“claims ladder”** in the thesis:

  1. **Strong (supported):** comparisons are fair; baselines explicit; backbone gap interpretable.  
  2. **Medium:** agent neighbors help/hurt relative to fixed heuristics *under your setting*.  
  3. **Weak / future:** economic profit, full walk-forward, broader universes.

- Add one paragraph: **“If MAE improves but economics are unknown, what we learned anyway.”**

---

## 3) Baselines beyond A/B/Proposed

### Likely questions

> “Why not a simpler neighbor selector?”  
> “How do we know the LLM adds value vs your metadata features alone?”

### What to answer (honest tiers)

1. **Minimum credible (often enough for MSc):**  
   - Target-only vs fixed-correlation neighbors vs best logged proposed set **under identical splits**.  
   - State clearly what each arm means and how “best proposed” is chosen from logs.

2. **Strengthen if asked / if time allows:**  
   - **Greedy / scoring baseline:** select top predictors by a *predefined score* (e.g., rolling correlation, sector heuristic) without LLM—same budget of neighbors.  
   - **Ablations:** no memory; no news/qualitative channel (if used); single-round proposals.  
   - **LLM vs non-LLM planner:** same tool APIs but rule-based policy (even simplistic) to show **where LLM changes exploration**, not just evaluation noise.

### How to present

- A **baseline table** in the thesis with **one sentence per baseline**: *what hypothesis it represents*.  
- If you can’t run everything, include a **Limitations** subsection listing the missing baseline and **why** it’s the next step.

---

## 4) Economic relevance (turnover, costs, capacity)

### Likely question

> “MAE on returns doesn’t mean a strategy makes money.”

### What to answer

- Agree quickly, then reframe: thesis demonstrates **signal discovery + measurement discipline**; **PnL is a downstream layer** requiring positions, costs, Borrow constraints (India-specific realities optional but impressive if mentioned briefly).
- Offer **interpretability bridges** without full backtest:
  - If forecasts change slowly, **turnover may be moderate** (hypothesis).  
  - If neighbor sets are unstable week-to-week, **capacity / implementability** suffers (honest).

### How to present

- Add a short **“Toward portfolio evaluation”** section: 5–10 lines + bullet roadmap (transaction costs, roll-forward, restrictions).  
- Optional: a **toy portfolio rule** (rank stocks by predicted return, equal-weight top-k, report turnover proxy) *only if you can do it cleanly*—not required if you label it exploratory.

---

## Slide / chapter checklist (defense-friendly)

Use this as a final pass before submission or viva prep.

- [ ] **Contribution sentence** on slide 2 / Chapter 1 end  
- [ ] **Evaluation contract** spelled out: alignment, purge/split, horizons, baselines  
- [ ] **One figure** that tells the backbone story without tables  
- [ ] **“Stateless LSTM”** panel: \(L\)-day window + “no hidden state across experiments” (see section above)  
- [ ] **One table** of baselines with plain-language interpretation  
- [ ] **Limitations** that read as competence, not apology  
- [ ] **Reproducibility pointer**: what artifacts exist (logs, configs, code)  
- [ ] **Claims ladder** (what you did / did not prove)

---

## “If they only ask one thing” (good closer)

Prepare a 30-second closing:

> “This thesis delivers a **repeatable experimentation pipeline** for neighbor discovery with **leakage-aware evaluation**. The main empirical takeaway is **model-dependent**: the same neighbor enrichment can help tree models substantially while adding less marginal value for the LSTM under our features—**and naive neighbor selection can be materially harmful**. The immediate next step is **economic validation** with costs and walk-forward testing.”

---

## File location

This guide lives at: `docs/masters_thesis_review_prep.md`

You can paste sections into your thesis **Discussion**, **Limitations**, or **Defense slides** verbatim (after tailoring to your examiner’s field: CS vs Finance vs Analytics).
