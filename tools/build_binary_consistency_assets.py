"""
Build binary-consistency aggregate assets:
  - tables/08_consistency_binary.tex
  - figures/binary_consistency_overview.pdf/png  (cross-model 4-panel comparison)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
EVAL_OUT  = ROOT / "vlm_eval_outputs"
TABLES    = ROOT / "PAPER_PLOTS" / "tables"
FIGS      = ROOT / "paper_assets" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

VLMs = [
    "internvl3_1b", "internvl3_2b", "internvl3_8b",
    "qwen2vl_2b",   "qwen25vl_3b",  "qwen25vl_7b",
    "llava_onevision_7b", "llava_13b", "llama32_11b",
]
DISPLAY = {
    "internvl3_1b":       "InternVL3-1B",
    "internvl3_2b":       "InternVL3-2B",
    "internvl3_8b":       "InternVL3-8B",
    "qwen2vl_2b":         "Qwen2-VL-2B",
    "qwen25vl_3b":        "Qwen2.5-VL-3B",
    "qwen25vl_7b":        "Qwen2.5-VL-7B",
    "llava_onevision_7b": "LLaVA-OV-7B",
    "llava_13b":          "LLaVA-1.6-13B",
    "llama32_11b":        "LLaMA-3.2-11B",
}

# ── Load metrics ──────────────────────────────────────────────────────────────
# vlm_consistency_eval.py uses different model keys for some models
# (e.g. "llava_ov_7b" instead of "llava_onevision_7b")
ALT_KEYS = {
    "llava_onevision_7b": "llava_ov_7b",
}

rows = []
for k in VLMs:
    candidates = [k] + ([ALT_KEYS[k]] if k in ALT_KEYS else [])
    for cand in candidates:
        p = EVAL_OUT / "binary" / cand / "consistency" / "consistency_metrics.json"
        if p.exists():
            with open(p) as f:
                rows.append((k, json.load(f)))
            break
    else:
        print(f"  [WARN] no consistency metrics for {k}")

print(f"Loaded {len(rows)} model consistency metrics for binary task")

# ── Build LaTeX table ─────────────────────────────────────────────────────────
def fmt(x, p=3):
    if x is None:
        return "--"
    return f"{float(x):.{p}f}"

def bold_best(values, lower_better=False):
    nums = [v for v in values if v is not None]
    if not nums:
        return [fmt(v) for v in values]
    best = min(nums) if lower_better else max(nums)
    out = []
    for v in values:
        if v is None:
            out.append("--")
        elif abs(float(v) - best) < 1e-9:
            out.append(rf"\textbf{{{fmt(v)}}}")
        else:
            out.append(fmt(v))
    return out

cols = [
    ("majority_vote_accuracy",     r"MV Acc.",                      False),
    ("majority_vote_balanced_acc", r"MV Bal.~Acc.",                 False),
    ("majority_vote_f1",           r"MV F1",                        False),
    ("mean_consistency",           r"Mean Consistency",             False),
    ("mean_entropy",               r"Mean Entropy $\downarrow$",    True),
    ("ece",                        r"ECE $\downarrow$",             True),
]
header = ["Model"] + [c[1] for c in cols]
formatted = {c[0]: bold_best([r[1].get(c[0]) for r in rows], lower_better=c[2]) for c in cols}

tex = []
tex.append(r"\begin{table*}[t]")
tex.append(r"\caption{Binary anomaly detection consistency under stochastic "
           r"decoding ($T=0.7$, $N{=}5$ runs per image). MV~Acc. = majority-vote "
           r"accuracy; Mean Consistency = fraction of runs agreeing with the "
           r"majority vote; Entropy = predictive entropy across runs; ECE = "
           r"expected calibration error of mean confidence.}")
tex.append(r"\label{tab:consistency_binary}")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\begin{tabular}{l" + "c" * len(cols) + "}")
tex.append(r"\toprule")
tex.append(" & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\")
tex.append(r"\midrule")
for i, (k, _) in enumerate(rows):
    cells = [DISPLAY[k]] + [formatted[c[0]][i] for c in cols]
    tex.append(" & ".join(cells) + r" \\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table*}")
(TABLES / "08_consistency_binary.tex").write_text("\n".join(tex) + "\n")
print(f"Saved {TABLES / '08_consistency_binary.tex'}")

# ── Cross-model comparison figure (4 panels) ──────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

labels = [DISPLAY[k] for k, _ in rows]
mv_acc   = [r[1]["majority_vote_accuracy"]      for r in rows]
mv_bacc  = [r[1]["majority_vote_balanced_acc"]  for r in rows]
mv_f1    = [r[1]["majority_vote_f1"]            for r in rows]
mean_c   = [r[1]["mean_consistency"]            for r in rows]
mean_e   = [r[1]["mean_entropy"]                for r in rows]
ece      = [r[1]["ece"]                         for r in rows]

# Sort by majority-vote balanced accuracy for stable ordering
order = np.argsort(mv_bacc)
labels_o = [labels[i] for i in order]

def metric_panel(ax, vals, title, lower_better=False):
    vals_o = [vals[i] for i in order]
    cmap = plt.cm.RdYlGn_r if lower_better else plt.cm.RdYlGn
    arr = np.array(vals_o)
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    bars = ax.barh(labels_o, vals_o, color=cmap(norm),
                    edgecolor="gray", linewidth=0.5)
    for bar, v in zip(bars, vals_o):
        ax.text(v + max(vals_o) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7, color="#333")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
metric_panel(axes[0, 0], mv_bacc, "Majority-Vote Balanced Accuracy",          False)
metric_panel(axes[0, 1], mean_c,  "Mean Consistency (fraction agree with MV)", False)
metric_panel(axes[1, 0], mean_e,  "Mean Entropy (lower is more consistent)",  True)
metric_panel(axes[1, 1], ece,     "ECE (lower is better calibrated)",          True)
fig.suptitle("Binary Anomaly Detection Consistency under Stochastic Decoding "
             "($T=0.7$, $N{=}5$ runs)", fontsize=11, y=1.01)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"binary_consistency_overview.{ext}")
plt.close(fig)
print(f"Saved {FIGS / 'binary_consistency_overview.pdf'}")
print(f"Saved {FIGS / 'binary_consistency_overview.png'}")
