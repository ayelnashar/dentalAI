# Aggregate the numbers reported in the paper from a results folder into nums.json.
# Usage: python aggregate_results.py <results_folder> [<out.json>]
# Needs the summary JSON/CSV files and per_tooth/ (for per-condition gate acceptance).
import json, os, sys, numpy as np, pandas as pd
R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "nums.json")
L = lambda f: json.load(open(os.path.join(R, f)))
CONDS = ["Impacted", "Caries", "Periapical Lesion"]
def m(xs):
    xs = [x for x in xs if x is not None and x == x]
    return (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, len(xs)) if xs else (None, None, 0)

def agg(rows, key):
    """rows: per-seed records with tgt/tgt_ext/src_before/src_after; returns aggregate dict"""
    o = {}
    for view in ["full", "src", "tgt", "cov50", "fix50", "cov20", "fix20"]:
        o[f"{view}_macro"] = m([r["tgt"][view].get("macro") for r in rows if view in r["tgt"]])
        o[f"{view}_cov"] = m([r["tgt"][view].get("coverage") for r in rows if view in r["tgt"]])
        for c in CONDS:
            o[f"{view}_{c}"] = m([r["tgt"][view].get(f"auc_{c}") for r in rows if view in r["tgt"]])
    o["full_ece"] = m([r["tgt"]["full"].get("ece") for r in rows])
    if all("tgt_ext" in r for r in rows):
        for view in ["full", "src", "tgt"]:
            o[f"ext_{view}_macro"] = m([r["tgt_ext"][view].get("macro") for r in rows])
            for c in CONDS: o[f"ext_{view}_{c}"] = m([r["tgt_ext"][view].get(f"auc_{c}") for r in rows])
    sb = [r.get("src_before", r.get("dentex_before")) for r in rows]
    if rows and "src_before" in rows[0]:
        o["dentex_before"] = m([r["src_before"]["macro4_full"] for r in rows])
        o["dentex_after"] = m([r["src_after"]["macro4_full"] for r in rows])
        drops = [r["src_before"]["macro4_full"] - r["src_after"]["macro4_full"] for r in rows]
        o["dentex_drop"] = m(drops); o["dentex_pass"] = int(sum(d <= 0.02 for d in drops)); o["n"] = len(rows)
        o["src_refusal_after"] = m([r["src_after"].get("refusal") for r in rows])
    o["delta_ok"] = all(r.get("delta_ok", True) for r in rows)
    if rows and "controls" in rows[0] and rows[0]["controls"]:
        for ck in rows[0]["controls"]:
            o[f"ctrl_{ck}_full_macro"] = m([r["controls"][ck]["full"].get("macro") for r in rows])
            for c in CONDS: o[f"ctrl_{ck}_full_{c}"] = m([r["controls"][ck]["full"].get(f"auc_{c}") for r in rows])
    return o

N = {}
B = L("B_frozen_ladder.json"); C = L("C_live_trunk.json")
groups = {}
for r in B: groups.setdefault(("B", r["backbone"], r["technique"], str(r["budget"])), []).append(r)
for r in C: groups.setdefault(("C", r["backbone"], r["method"], str(r["budget"])), []).append(r)
LAD = {}
for k, rows in groups.items():
    a = agg(rows, k)
    a["params"] = int(rows[0].get("adapt_params", rows[0].get("trainable", 0)))
    if k[0] == "C":
        a["train_min"] = m([r["train_minutes"] for r in rows]); a["total_min"] = m([r["total_minutes"] for r in rows])
        a["mem_gb"] = m([r.get("mps_driver_alloc_gb") for r in rows])
    LAD["|".join(k)] = a
N["ladder"] = LAD
# parameter counts from E5 (adaptation params)
for f in ["E5_table_B.csv", "E5_table_C.csv"]:
    t = pd.read_csv(os.path.join(R, f))
    for _, row in t.iterrows():
        cell = "B" if "B.csv" in f else "C"
        k = f"{cell}|{row.backbone}|{row.technique}|{row.budget}"
        if k in LAD: LAD[k]["params_adaptation"] = int(row.params_adaptation) if row.params_adaptation == row.params_adaptation else None

# D1 regimes
D1 = L("D1_joint_budgets.json"); REG = {}
for r in D1:
    k = f"{r['backbone']}|{r['regime']}|{r['budget']}"
    REG.setdefault(k, []).append(r)
N["regimes"] = {}
for k, rows in REG.items():
    o = {"full_macro": m([r["tgt"]["full"]["macro"] for r in rows]),
         "src_macro": m([r["tgt"]["src"].get("macro") for r in rows]),
         "src_cov": m([r["tgt"]["src"]["coverage"] for r in rows]),
         "ext_macro": m([r.get("tgt_ext", {}).get("full", {}).get("macro") for r in rows]),
         "dentex": m([r["dentex"]["macro4_full"] for r in rows])}
    for c in CONDS: o[c] = m([r["tgt"]["full"][f"auc_{c}"] for r in rows])
    N["regimes"][k] = o

# D2 CTT
d2 = pd.read_csv(os.path.join(R, "D2_ctt.csv")); d2["b"] = d2.budget.astype(str)
CT = {}
for (dirn, bb, cond, bud), g in d2.groupby(["direction", "backbone", "condition", "b"]):
    cis = [json.loads(x) if isinstance(x, str) and x.startswith("[") else None for x in g.R_ci95]
    lo = [c[0] for c in cis if c]; hi = [c[1] for c in cis if c]
    CT[f"{dirn}|{bb}|{cond}|{bud}"] = {
        "A_S2T": m(g.A_S2T), "A_T": m(g.A_T), "G": m(g.G), "R": m(g.R), "Rmin": float(g.R.min()), "Rmax": float(g.R.max()),
        "ci_lo_mean": float(np.mean(lo)) if lo else None, "ci_hi_mean": float(np.mean(hi)) if hi else None,
        "verdicts": list(g.verdict), "P_ge_rho": m(g.P_R_ge_rho),
        "R_out": m(g.R_outcome) if "R_outcome" in g else None, "outcomes": [str(x) for x in g.outcome],
        "agree": [bool(x) if x == x else None for x in g.agree], "A_T_test": m(g.A_T_test), "A_protocol_test": m(g.A_protocol_test),
        "R_out_matched": m(g.R_outcome_matched_budget)}
N["ctt"] = CT
dd = d2[d2.direction == "D2A"]
N["ctt_agree"] = {b: [int((dd[dd.b == b].agree == True).sum()), int(dd[dd.b == b].agree.notna().sum())] for b in ["64", "256", "all"]}
# rho sensitivity at 256: verdict changes vs rho = 0.5, over all D2A seed-level rows
x = dd[dd.b == "256"]
N["ctt_rho"] = {}
for rho in [0.3, 0.4, 0.45, 0.55, 0.6, 0.7]:
    v0 = x.R >= 0.5; v1 = x.R >= rho
    N["ctt_rho"][str(rho)] = int((v0 != v1).sum())
N["ctt_rows_256"] = int(len(x))
# D3 geometry
d3 = pd.read_csv(os.path.join(R, "D3_geometry_baseline.csv"))
N["geometry"] = {f"{s}|{c}": m(g.auroc) for (s, c), g in d3.groupby(["setting", "condition"])}
# gate matrix
e6 = pd.read_csv(os.path.join(R, "E6_gate_matrix.csv"))
N["gate"] = {f"{r.cell}|{r.backbone}|{r.technique}|{r.budget}": {k: float(getattr(r, k)) for k in
             ["srcgate_on_DENTEX", "srcgate_on_ADCD", "tgtgate_on_ADCD", "tgtgate_on_DENTEX", "srcgate_on_ADCDext", "tgtgate_on_ADCDext"]}
             for r in e6.itertuples()}
# per-condition acceptance from per-tooth files
ACC = {}
import glob
for f in glob.glob(os.path.join(R, "per_tooth", "*_s42.npz")):
    tag = os.path.basename(f)[:-8]
    if not (tag.startswith("B_") or tag.startswith("C_")): continue
    sa, ta = [], []
    for s in (42, 43, 44):
        p = os.path.join(R, "per_tooth", f"{tag}_s{s}.npz")
        if not os.path.exists(p): continue
        z = np.load(p); y = z["labels"]; a = z["maha_src"] <= z["tau_src"]; at = z["maha_tgt"] <= z["tau_tgt"]
        sa.append([float(a[y[:, j] == 1].mean()) for j in range(3)]); ta.append([float(at[y[:, j] == 1].mean()) for j in range(3)])
    ACC[tag] = {"src": list(np.mean(sa, 0)), "tgt": list(np.mean(ta, 0))}
N["accept_by_condition"] = ACC
# coverage curves
e4 = pd.read_csv(os.path.join(R, "E4_auroc_coverage.csv"))
N["coverage_curves"] = {f"{c}|{b}|{t}|{bu}": [[float(r.coverage), (float(r.macro_mean) if r.macro_mean == r.macro_mean else None), int(r.n_seeds_defined)] for r in g.itertuples()]
                        for (c, b, t, bu), g in e4.groupby(["cell", "backbone", "technique", "budget"])}
# E1 CIs
e1 = pd.read_csv(os.path.join(R, "E1_auroc_ci.csv"))
N["ci"] = {f"{r.cell}|{r.backbone}|{r.technique}|{r.budget}|{r.view}|{r.condition}": [float(r.auroc_seed_mean), float(r.ci_lo), float(r.ci_hi)] for r in e1.itertuples()}
e2 = pd.read_csv(os.path.join(R, "E2_paired_diff.csv"))
N["paired"] = {f"{r.cell}|{r.backbone}|{r.technique}|{r.budget}|{r.view}|{r.condition}": [float(r.diff_vs_source_only), float(r.ci_lo), float(r.ci_hi), bool(r.excludes_zero)] for r in e2.itertuples()}
# fusion
F = L("B2_fusion.json"); FU = {}
for r in F: FU.setdefault(r["technique"], []).append(r)
N["fusion"] = {k: {"full_macro": m([r["tgt"]["full"]["macro"] for r in v]), "src_macro": m([r["tgt"]["src"].get("macro") for r in v]),
                   "ext_macro": m([r.get("tgt_ext", {}).get("full", {}).get("macro") for r in v]),
                   "Caries": m([r["tgt"]["full"]["auc_Caries"] for r in v]), "Impacted": m([r["tgt"]["full"]["auc_Impacted"] for r in v]),
                   "Periapical Lesion": m([r["tgt"]["full"]["auc_Periapical Lesion"] for r in v]),
                   "dentex": m([r["dentex"]["macro4_full"] for r in v]) if "dentex" in v[0] else None} for k, v in FU.items()}
# dataset
N["flow"] = pd.read_csv(os.path.join(R, "dataset_flow_v2.csv")).fillna("").to_dict("records")
N["parse"] = L("adcd_parse_stats.json"); N["crop"] = L("crop_size_stats.json"); N["env"] = L("env.json")
N["emb_min"] = L("embedding_minutes.json"); N["G"] = L("G_healthy_teeth_status.json")
_ds = os.path.join(R, "dupsens.json")
N["dupsens"] = json.load(open(_ds)) if os.path.exists(_ds) else {}
N["max_dev_all"] = float(max([r["tgt"]["max_dev_tgt"] for r in B + C] + [r["src_after"]["max_dev"] for r in B + C]))
N["n_runs"] = {"B": len(B), "C": len(C), "B2": len(F), "D1": len(D1), "D2": int(len(d2))}
json.dump(N, open(OUT, "w"), indent=1, default=float)
print("ok", OUT, "max dev", N["max_dev_all"], N["ctt_agree"], N["ctt_rho"])
