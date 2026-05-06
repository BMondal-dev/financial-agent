#!/usr/bin/env python3
"""Generate the full 56-target baseline comparison LaTeX table."""
import json

with open("services/data/analysis_h10.json") as f:
    d = json.load(f)

bc = d["baseline_comparison"]

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\scriptsize")
lines.append(r"\caption{Complete Baseline A vs B vs Proposed comparison at horizon $H=10$ for all 56 NIFTY constituents (XGB backbone in \texttt{analysis\_h10.json}). Lower MAE is better. Winner: A=target-only, B=fixed correlation, Proposed=agent-proposed.}")
lines.append(r"\label{tab:full-baseline-xgb}")
lines.append(r"\begin{tabular}{lccccccccc}")
lines.append(r"\toprule")
lines.append(r"\# & Target & Sector & A MAE & B MAE & Prop.\ MAE & Winner & A B0 & B B0 & Prop.\ B0 \\")
lines.append(r"\midrule")

for i, pt in enumerate(bc["per_target"], 1):
    t = pt["target"]
    w = pt["winner"]
    a = round(pt["A_mae"], 4) if pt["A_mae"] else r"---"
    b = round(pt["B_mae"], 4) if pt["B_mae"] else r"---"
    o = round(pt["Ours_mae"], 4) if pt["Ours_mae"] else r"---"
    ab = r"$\checkmark$" if pt["A_beats_0"] else r"$\times$"
    bb = r"$\checkmark$" if pt["B_beats_0"] else r"$\times$"
    ob = r"$\checkmark$" if pt["Ours_beats_0"] else r"$\times$"
    sec = pt["sector"]
    lines.append(f"{i} & {t} & {sec} & {a} & {b} & {o} & \\textbf{{{w}}} & {ab} & {bb} & {ob} \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

print("\n".join(lines))
