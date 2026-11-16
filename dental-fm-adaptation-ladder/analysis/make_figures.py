# Figures 2-9 of the paper from nums.json (see aggregate_results.py).
# Usage: python make_figures.py <results_folder> <figure_folder> [<nums.json>]
# Figure 2 needs Figure_crop_examples.png from the notebook run; it is skipped if that file is absent.
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
R = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
N = json.load(open(sys.argv[3] if len(sys.argv) > 3 else os.path.join(R, "nums.json"))); LAD = N["ladder"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": True,
                     "axes.spines.right": True, "axes.edgecolor": "#3a3936", "axes.linewidth": 0.7,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "savefig.dpi": 300})
S1, S2, S3, S4, S5, S6, S7 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
INK2, MUTED = "#52514e", "#8a8984"
COND = ["Impacted", "Caries", "Periapical Lesion"]; CLAB = ["Impacted", "Caries", "Periapical"]; CCOL = [S1, S2, S3]
BBL = {"dental_vitl": "Dental ViT-L", "dinov3": "DINOv3", "biomedclip": "BiomedCLIP"}
NICE = {"source_only": "Source only", "R0_thresh": "Threshold recal.", "R0_moment": "Feature alignment", "E_bias": "Emb. bias",
        "E_lowrank_r2": "Emb. low-rank r2", "E_lowrank_r4": "Emb. low-rank r4", "E_lowrank_r8": "Emb. low-rank r8", "E_lowrank_r16": "Emb. low-rank r16",
        "E_linear_l2": "Emb. linear L2", "E_linear_replay": "Emb. linear replay", "E_linear_avg": "Emb. linear avg", "R1_head": "Head re-fit",
        "lora": "LoRA", "adapter": "Adapter", "vpt": "VPT", "bitfit": "BitFit", "ln_affine": "LN-affine",
        "full_ewc": "Full + EWC", "full_rehearsal": "Full + rehearsal", "full_featanchor": "Full + feat. anchor"}
def rung(t):
    if t in ("source_only",): return "src"
    if t.startswith("R0"): return "R0"
    if t == "R1_head": return "R1"
    if t.startswith("E_"): return "E"
    if t.startswith("full"): return "R3"
    return "R2"
RSTY = {"src": ("Source only", "#0b0b0b", "o"), "R0": ("Rung 0 label-free", MUTED, "s"), "R1": ("Rung 1 head re-fit", S2, "D"),
        "E": ("Embedding-space", S4, "v"), "R2": ("Rung 2 live PEFT", S1, "^"), "R3": ("Rung 3 full trunk", S3, "P")}
def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", facecolor="white"); fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 2: crop examples (two per row from the notebook figure, arranged 2 x 6) ----------------
_crop_png = os.path.join(R, "Figure_crop_examples.png")
HAVE_CROPS = os.path.exists(_crop_png)
im = Image.open(_crop_png).convert("RGB") if HAVE_CROPS else None; a = np.asarray(im.convert("L")) if HAVE_CROPS else None
bands = [(81, 579), (674, 1172), (1266, 1764), (1859, 2357), (2451, 2950), (3044, 3542)]
def panels(y0, y1):
    cols = (a[y0:y1] < 250).sum(0) > 20; segs = []; inb = False
    for x, v in enumerate(cols):
        if v and not inb: s = x; inb = True
        elif not v and inb: segs.append((s, x)); inb = False
    if inb: segs.append((s, len(cols)))
    segs = [g for g in segs if g[1] - g[0] > 60]
    return [im.crop((x0, y0, x1, y1)) for x0, x1 in segs]
P = [panels(*b) for b in bands] if HAVE_CROPS else None   # rows: DENTEX imp, ADCD imp, DENTEX car, ADCD car, DENTEX per, ADCD per
fig, ax = plt.subplots(2, 6, figsize=(7.16, 2.9))
for ci, cname in enumerate(CLAB if HAVE_CROPS else []):
    for di, dname in enumerate(["DENTEX", "ADCD"]):
        row = P[2 * ci + di]
        for k in range(2):
            pim = row[k].copy(); W0, H0 = pim.size
            from PIL import ImageDraw
            ImageDraw.Draw(pim).rectangle([0, 0, W0 - 1, H0 - 1], outline=(40, 40, 40), width=max(2, W0 // 120))
            Wt = int(round(H0 * 0.92))
            if W0 < Wt:
                canvas = Image.new("RGB", (Wt, H0), "white"); canvas.paste(pim, ((Wt - W0) // 2, 0)); pim = canvas
            else:
                Ht = int(round(W0 / 0.92)); canvas = Image.new("RGB", (W0, Ht), "white"); canvas.paste(pim, (0, (Ht - H0) // 2)); pim = canvas
            A = ax[di, 2 * ci + k]; A.imshow(pim); A.set_xticks([]); A.set_yticks([])
            for sp in A.spines.values(): sp.set_visible(False)   # frame is drawn on the crop itself
            if k == 0: A.set_title(f"{dname}: {cname}", loc="left", fontsize=7.5, color="#0b0b0b")
fig.subplots_adjust(wspace=0.08, hspace=0.18)
if HAVE_CROPS: save(fig, "Fig2_crops")
else: plt.close(fig); print("Figure 2 skipped (Figure_crop_examples.png not found)")

# ---------------- Fig 3: target AUROC versus source change, one panel per backbone ----------------
TOL = 0.02; XL = -0.05
fig, axs = plt.subplots(3, 1, figsize=(3.5, 7.4), sharex=True, sharey=True)
LABS = {"dental_vitl": {"source_only": (0.006, 0.772), "R1_head": (-0.0465, 0.895), "full_rehearsal": (0.006, 0.955), "full_ewc": (-0.040, 0.945),
                        "full_featanchor": (0.010, 0.889), "adapter": (0.012, 0.842), "E_linear_replay": (-0.046, 0.858), "lora": (0.012, 0.812)},
        "dinov3": {"source_only": (-0.046, 0.752), "R1_head": (-0.0465, 0.880), "full_rehearsal": (-0.012, 0.935), "full_ewc": (0.0135, 0.835),
                   "full_featanchor": (0.0135, 0.785), "adapter": (0.010, 0.697), "E_lowrank_r8": (-0.046, 0.712)},
        "biomedclip": {"source_only": (0.004, 0.676), "R1_head": (-0.0465, 0.890), "E_lowrank_r8": (-0.046, 0.598)}}
SHORT = {"source_only": "Source only", "R1_head": "Head re-fit", "full_rehearsal": "Full + rehearsal", "full_ewc": "Full + EWC",
         "full_featanchor": "Full + anchor", "adapter": "Adapter", "E_linear_replay": "Emb. linear replay", "lora": "LoRA", "E_lowrank_r8": "Emb. corrections"}
for A, bb, pl in zip(axs, ["dental_vitl", "dinov3", "biomedclip"], "abc"):
    pts = []
    for k, v in LAD.items():
        cell, b, t, bud = k.split("|")
        if b != bb or bud not in ("0", "256") or t == "R0_thresh": continue
        pts.append((t, v["dentex_after"][0] - v["dentex_before"][0], v["full_macro"][0]))
    A.axvspan(-1, -TOL, color="#f3f2ef", zorder=0); A.axvline(-TOL, color=MUTED, lw=0.8, ls="--")
    ref = N["regimes"][f"{bb}|target_only|256"]["full_macro"][0]; A.axhline(ref, color=MUTED, lw=0.8, ls=":")
    for t, x, y in pts:
        lab, col, mk = RSTY[rung(t)]
        if x < XL:
            A.plot(XL, y, "<", ms=6, color=col, mec="white", mew=0.6, zorder=3, clip_on=False)
        else:
            A.plot(x, y, mk, ms=5, color=col, mec="white", mew=0.6, zorder=3)
    for t, x, y in pts:
        if t in LABS[bb]:
            tx, ty = LABS[bb][t]; txt = SHORT[t]
            if t == "R1_head": txt = f"Head re-fit (change {x:+.2f})"
            A.annotate(txt, (max(x, XL), y), xytext=(tx, ty), fontsize=6.0, color=INK2,
                       arrowprops=dict(arrowstyle="-", color="#b5b4ae", lw=0.5, shrinkA=0, shrinkB=3))
    A.set_title(f"({pl}) {BBL[bb]}", loc="left"); A.set_xlim(XL, 0.035); A.set_xticks([-0.04, -0.02, 0.0, 0.02]); A.grid(True, axis="y")
    A.set_ylim(0.58, 0.97); A.set_ylabel("ADCD macro AUROC")
axs[-1].set_xlabel("DENTEX macro AUROC change")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=RSTY[r][2], color=RSTY[r][1], ls="", ms=5, mec="white", label=RSTY[r][0]) for r in ["src", "R0", "E", "R1", "R2", "R3"]]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.subplots_adjust(hspace=0.22, bottom=0.12); save(fig, "Fig3_tradeoff")

# ---------------- Fig 4: per-condition AUROC with image-level CIs, Dental ViT-L ----------------
rows = [("B", "source_only", "0"), ("B", "R0_moment", "0"), ("B", "E_lowrank_r16", "256"), ("B", "E_linear_replay", "256"),
        ("C", "vpt", "256"), ("C", "bitfit", "256"), ("C", "ln_affine", "256"), ("C", "lora", "256"), ("C", "adapter", "256"),
        ("C", "full_ewc", "256"), ("C", "full_featanchor", "256"), ("C", "full_rehearsal", "256"), ("B", "R1_head", "256")]
fig, A = plt.subplots(figsize=(3.5, 3.6))
for i, (cell, t, bud) in enumerate(rows):
    for j, c in enumerate(COND):
        mu, lo, hi = N["ci"][f"{cell}|dental_vitl|{t}|{bud}|full|{c}"]
        yy = i + (j - 1) * 0.24
        A.plot([lo, hi], [yy, yy], color=CCOL[j], lw=1.0); A.plot(mu, yy, "o", ms=3.6, color=CCOL[j], mec="white", mew=0.4)
for j, c in enumerate(COND):
    ref = N["ci"][f"D|dental_vitl|target_only|all|full|{c}"][0]
    A.axvline(ref, color=CCOL[j], lw=0.6, ls=":")
A.set_yticks(range(len(rows))); A.set_yticklabels([NICE[t] + (" (256)" if t == "R1_head" else "") for _, t, _ in rows]); A.invert_yaxis()
A.set_xlabel("ADCD AUROC, full coverage (95% image-level CI)"); A.grid(True, axis="x"); A.set_xlim(0.66, 1.0)
h = [Line2D([], [], marker="o", color=CCOL[j], ls="-", ms=3.6, label=CLAB[j]) for j in range(3)]
A.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.4, -0.2))
save(fig, "Fig5_per_condition")

# ---------------- Fig 5: label budget, Dental ViT-L ----------------
ALLN = np.mean([r["crops"] for r in N["flow"] if r["dataset"] == "ADCD" and str(r["split"]).startswith("train_seed")])
def xb(b): return ALLN if b == "all" else float(b)
fig, axs = plt.subplots(2, 1, figsize=(3.5, 5.0), sharex=True)
ser = [("Head re-fit (target only)", S2, "D", [("B", "R1_head", b) for b in ("16", "32", "64", "128", "256")] + [("D", "target_only", "all")]),
       ("Joint", S1, "o", [("D", "joint", b) for b in ("16", "64", "256", "all")]),
       ("Joint, domain-balanced", S7, "s", [("D", "joint_balanced", b) for b in ("16", "64", "256", "all")]),
       ("Emb. low-rank r8", S4, "v", [("B", "E_lowrank_r8", b) for b in ("16", "32", "64", "128", "256")])]
for k, (lab, col, mk, pts) in enumerate(ser):
    for ai, key in enumerate(["full_macro", "dentex"]):
        xs, ys = [], []
        for cell, t, b in pts:
            if cell == "D": v = N["regimes"][f"dental_vitl|{t}|{b}"]; y = v["full_macro" if ai == 0 else "dentex"][0]
            else:
                v = LAD[f"B|dental_vitl|{t}|{b}"]; y = v["full_macro"][0] if ai == 0 else v["dentex_after"][0]
            xs.append(xb(b)); ys.append(y)
        axs[ai].plot(xs, ys, marker=mk, color=col, lw=1.6, ms=4.5, mec="white", mew=0.5, label=lab)
fr = LAD["C|dental_vitl|full_rehearsal|256"]; fe = LAD["C|dental_vitl|full_ewc|256"]
for ai in range(2):
    axs[ai].plot(256, fr["full_macro"][0] if ai == 0 else fr["dentex_after"][0], "P", ms=8, color=S3, mec="white", mew=0.6, label="Full + rehearsal", zorder=5)
    src = LAD["B|dental_vitl|source_only|0"]
    axs[ai].axhline(src["full_macro"][0] if ai == 0 else src["dentex_before"][0], color=MUTED, lw=0.8, ls=":")
    axs[ai].set_xscale("log"); axs[ai].set_xticks([16, 32, 64, 128, 256, ALLN]); axs[ai].set_xticklabels(["16", "32", "64", "128", "256", f"all\n(~{int(round(ALLN, -2)):,})"])
    axs[ai].minorticks_off(); axs[ai].grid(True, axis="y")
axs[0].set_ylabel("ADCD macro AUROC"); axs[1].set_ylabel("DENTEX macro AUROC"); axs[1].set_xlabel("Labeled ADCD crops")
axs[0].set_title("(a) ADCD test split", loc="left"); axs[1].set_title("(b) DENTEX test split", loc="left")
axs[1].axhspan(0.6, src["dentex_before"][0] - TOL, color="#f3f2ef", zorder=0); axs[1].set_ylim(0.66, 0.92)
axs[0].text(17, src["full_macro"][0] + 0.004, "source only", fontsize=6.3, color=INK2); axs[1].text(17, src["dentex_before"][0] + 0.004, "before adaptation", fontsize=6.3, color=INK2)
axs[1].text(17, src["dentex_before"][0] - TOL - 0.012, "below tolerance", fontsize=6.3, color=INK2)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.03))
axs[0].set_ylim(0.78, 0.98); fig.subplots_adjust(hspace=0.22, bottom=0.2); save(fig, "Fig6_label_budget")

# ---------------- Fig 6: cost axis ----------------
fig, axs = plt.subplots(2, 1, figsize=(3.5, 5.4), sharex=True, sharey=True)
for A, bb, pl in zip(axs, ["dental_vitl", "dinov3"], "ab"):
    for k, v in LAD.items():
        cell, b, t, bud = k.split("|")
        if b != bb or bud not in ("256",) or t == "R1_head": continue
        p = v.get("params_adaptation") or v["params"]
        if not p: continue
        lab, col, mk = RSTY[rung(t)]
        A.plot(p, v["full_macro"][0], mk, ms=5.5, color=col, mec="white", mew=0.5, zorder=3)
        if f"ctrl_tgt_head_full_macro" in v and v["ctrl_tgt_head_full_macro"][0] is not None and cell == "C":
            A.plot(p, v["ctrl_tgt_head_full_macro"][0], mk, ms=5.5, mfc="white", mec=col, mew=1.0, zorder=2)
            A.plot([p, p], [v["full_macro"][0], v["ctrl_tgt_head_full_macro"][0]], color=col, lw=0.5, alpha=0.6)
        OFF = {"dental_vitl": {"vpt": (0, -11, "center"), "ln_affine": (-1, -11, "right"), "bitfit": (1, -11, "left"), "lora": (5, -9, "left"), "adapter": (5, -9, "left"),
                               "full_ewc": (-7, -9, "right"), "full_rehearsal": (-7, 3, "right"), "full_featanchor": (-7, -2, "right"), "E_linear_replay": (-5, 3, "right")},
               "dinov3": {"vpt": (0, -11, "center"), "ln_affine": (-1, -11, "right"), "bitfit": (1, -11, "left"), "lora": (5, -9, "left"), "adapter": (6, 1, "left"),
                          "full_ewc": (-7, 1, "right"), "full_rehearsal": (-7, -2, "right"), "full_featanchor": (-7, -7, "right")}}
        SH = {"vpt": "VPT", "ln_affine": "LN-aff.", "bitfit": "BitFit", "lora": "LoRA", "adapter": "Adapter", "full_ewc": "Full + EWC",
              "full_rehearsal": "Full + rehearsal", "full_featanchor": "Full + anchor", "E_linear_replay": "Emb. linear replay"}
        if t in OFF[bb]:
            A.annotate(SH[t], (p, v["full_macro"][0]), xytext=OFF[bb][t][:2], textcoords="offset points", fontsize=5.8, color=INK2, ha=OFF[bb][t][2])
    src = LAD[f"B|{bb}|source_only|0"]["full_macro"][0]; A.axhline(src, color=MUTED, lw=0.8, ls=":")
    A.text(1.2e7, src - 0.011, "source only", fontsize=6.0, color=INK2)
    A.set_xscale("log"); A.set_title(f"({pl}) {BBL[bb]}", loc="left"); A.grid(True, axis="y"); A.set_ylabel("ADCD macro AUROC")
    A.set_xlim(5e2, 2e9)
axs[-1].set_xlabel("Trainable adaptation parameters")
h = [Line2D([], [], marker="o", color=INK2, ls="", ms=5, label="Source-fitted envelope"),
     Line2D([], [], marker="o", mfc="white", mec=INK2, ls="", ms=5, label="Head trained during adaptation (control)")] + \
    [Line2D([], [], marker=RSTY[r][2], color=RSTY[r][1], ls="", ms=5, label=RSTY[r][0]) for r in ["E", "R2", "R3"]]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(hspace=0.2, bottom=0.17); save(fig, "Fig4_cost_controls")

# ---------------- Fig 7: transferability ratio (two directions, stacked) ----------------
fig, axs = plt.subplots(2, 1, figsize=(3.5, 5.6), sharex=True)
for A, d, title in zip(axs, ["D2A", "A2D"], ["(a) DENTEX to ADCD", "(b) ADCD to DENTEX"]):
    ylab = []
    for i, bb in enumerate(["dental_vitl", "dinov3", "biomedclip"]):
        for j, c in enumerate(COND):
            v = N["ctt"][f"{d}|{bb}|{c}|256"]; y = i * 3.6 + j
            A.plot([v["ci_lo_mean"], v["ci_hi_mean"]], [y, y], color=CCOL[j], lw=1.0, clip_on=True)
            A.plot(v["R"][0], y, "o", ms=4.5, color=CCOL[j], mec="white", mew=0.5)
            ylab.append((y, f"{BBL[bb]}  {CLAB[j]}"))
    A.axvline(0.5, color=MUTED, ls="--", lw=0.8); A.axvline(0, color="#d0cfca", lw=0.6)
    A.set_title(title, loc="left"); A.grid(True, axis="x"); A.set_xlim(-0.6, 1.1)
    A.set_yticks([y for y, _ in ylab]); A.set_yticklabels([l for _, l in ylab], fontsize=6.3); A.set_ylim(9.9, -0.8)
    A.text(0.52, -0.25, "ρ = 0.5", fontsize=6.3, color=INK2)
axs[-1].set_xlabel("Transferability ratio R (development partition, 256 labels)")
fig.subplots_adjust(hspace=0.2); save(fig, "Fig7_transferability")

# ---------------- Fig 8: gate acceptance by condition ----------------
sel = [("B_dental_vitl_source_only_n0", "Source\nonly"), ("C_dental_vitl_adapter", "Adapter"),
       ("C_dental_vitl_full_ewc", "Full +\nEWC"), ("C_dental_vitl_full_rehearsal", "Full +\nrehearsal"),
       ("B_dinov3_source_only_n0", "Source\nonly"), ("C_dinov3_full_rehearsal", "Full +\nrehearsal")]
fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.9), sharex=True, sharey=True)
w = 0.26
for A, g, title in zip(axs, ["src", "tgt"], ["(a) Source-fitted gate", "(b) Gate re-fitted on unlabeled ADCD"]):
    for i, (tag, lab) in enumerate(sel):
        acc = N["accept_by_condition"][tag][g]
        for j in range(3):
            A.bar(i + (j - 1) * w, acc[j], width=w - 0.03, color=CCOL[j], label=CLAB[j] if i == 0 else None)
    A.set_xticks(range(len(sel))); A.set_xticklabels([l for _, l in sel], fontsize=6); A.set_ylim(0, 1.0); A.grid(True, axis="y")
    A.set_title(title, loc="left"); A.axvline(3.5, color="#d0cfca", lw=0.6)
    A.set_ylabel("Fraction accepted")
axs[-1].text(1.5, -0.30, "Dental ViT-L", ha="center", fontsize=6.8, color="#0b0b0b", transform=axs[-1].get_xaxis_transform())
axs[-1].text(4.5, -0.30, "DINOv3", ha="center", fontsize=6.8, color="#0b0b0b", transform=axs[-1].get_xaxis_transform())
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CCOL[j]) for j in range(3)], labels=CLAB, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.subplots_adjust(hspace=0.25, bottom=0.14, top=0.92); save(fig, "Fig8_gate_acceptance")

# ---------------- Fig 9: AUROC-coverage ----------------
fig, A = plt.subplots(figsize=(3.5, 2.6))
cur = [("B|dental_vitl|source_only|0", "Source only", "#0b0b0b", "o"), ("B|dental_vitl|E_linear_replay|256", "Emb. linear replay", S4, "v"),
       ("C|dental_vitl|adapter|256", "Adapter", S1, "^"), ("C|dental_vitl|full_ewc|256", "Full + EWC", S7, "P"),
       ("C|dental_vitl|full_rehearsal|256", "Full + rehearsal", S3, "X"), ("B|dental_vitl|R1_head|256", "Head re-fit (256)", S2, "D")]
for k, lab, col, mk in cur:
    pts = [(c, v) for c, v, n in N["coverage_curves"][k] if n == 3 and v is not None]
    A.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.6, label=lab, marker=mk, ms=3, markevery=2, mec="white", mew=0.4)
A.set_xlabel("Coverage (fraction of ADCD test crops, ordered by Mahalanobis distance)"); A.set_ylabel("Macro AUROC")
A.set_xlim(0.2, 1.02); A.grid(True, axis="y")
A.legend(frameon=False, fontsize=6.2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.24))
save(fig, "Fig9_auroc_coverage")
print("figures written to", OUT)
