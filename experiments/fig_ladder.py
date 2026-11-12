# Figure: the GraphNull null ladder on 362 held-out commits.
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "ladder_results.json"))["rows"]
OUT = sys.argv[2] if len(sys.argv) > 2 else "figs"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def ci(x, seed=20261108):
    x = np.asarray(x); bs = x[np.random.default_rng(seed).integers(0, len(x), (10000, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
levels = [("random", "Random graph\nsame edge count"), ("degree", "Degree-preserving\nrewire"),
          ("stratified", "Degree and file\nlocality preserved"), ("real", "Real call\ngraph")]
colors = {"random": "#d9d9d9", "degree": "#a6a6a6", "stratified": "#6b7fa8", "real": "#1f3b73"}
hatch = {"random": "", "degree": "///", "stratified": "", "real": ""}
fig, ax = plt.subplots(figsize=(6.8, 3.4))
w = 0.38
for j, h in enumerate((1, 2)):
    for i, (k, lab) in enumerate(levels):
        m, lo, hi = ci([r[f"{k}_{h}"] for r in R])
        x = i + (j - 0.5) * w
        ax.bar(x, m, w * 0.92, color=colors[k], hatch=hatch[k], edgecolor="#333333", linewidth=0.6,
               alpha=1.0 if h == 1 else 0.55)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", capsize=2.5, lw=0.8)
        ax.text(x, hi + 1.0, f"{m:.1f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(levels)))
ax.set_xticklabels([l for _, l in levels], fontsize=9)
ax.set_ylabel("Co-edit recall (%)", labelpad=8)
ax.set_ylim(0, 47)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#777777", edgecolor="#333333", label="1-hop neighbours"),
                   Patch(facecolor="#777777", alpha=0.55, edgecolor="#333333", label="2-hop neighbours")],
          loc="upper left", frameon=False, fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_ladder.{ext}", dpi=600)
print("figure written")
