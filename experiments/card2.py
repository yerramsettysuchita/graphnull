import random, sys, numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graphnull as gn
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
NAVY, MID, GRAY, LIGHT, INK = "#1f3b73", "#6b7fa8", "#9a9a9a", "#c8c8c8", "#111111"
# toy graph: 3 files (groups), mostly within-file calls
rng = random.Random(3)
groups = {n: n // 8 for n in range(24)}
pairs = set()
for _ in range(34):
    u = rng.randrange(24)
    v = rng.choice([x for x in range(24) if x != u and (groups[x] == groups[u] or rng.random() < 0.10)])
    pairs.add((min(u, v), max(u, v)))
G = nx.Graph(); G.add_nodes_from(range(24)); G.add_edges_from(pairs)
pos = {}
centers = [(-1.15, 0.7), (1.15, 0.7), (0.0, -1.15)]
for n in range(24):
    cx, cy = centers[groups[n]]; a = 2 * np.pi * (n % 8) / 8
    pos[n] = (cx + 0.5 * np.cos(a), cy + 0.5 * np.sin(a))
S, _ = gn.rewire_stratified(pairs, lambda x: groups[x], 7)
D, _ = gn.rewire_degree(pairs, 7)
assert gn.check_null(pairs, S, lambda x: groups[x])["group_profile_preserved"]
panels = [(pairs, NAVY, "real"), (S, MID, "file kept"), (D, GRAY, "degree kept")]
fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
fig.text(0.05, 0.80, "GraphNull", fontsize=23, color=INK, ha="left", va="center")
fig.text(0.05, 0.61, "Interface-matched null worlds\nfor code-graph tools", fontsize=10.5, color="#333333",
         ha="left", va="center", linespacing=1.3)
fig.text(0.05, 0.30, "Same tools, nodes and degrees.\nChoose what to scramble.", fontsize=8.5, color="#555555",
         ha="left", va="center", linespacing=1.35, style="italic")
for k, (P, col, lab) in enumerate(panels):
    ax = fig.add_axes([0.43 + k * 0.19, 0.12, 0.18, 0.72]); ax.set_axis_off()
    H = nx.Graph(); H.add_nodes_from(range(24)); H.add_edges_from(P)
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color=col, width=0.7)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=[8 + 5 * H.degree(n) for n in H], node_color="white",
                           edgecolors=INK, linewidths=0.5)
    ax.set_title(lab, fontsize=8.5, color=INK, pad=2)
    ax.set_aspect("equal")
for name, dpi in (("card2_1120x560", 200), ("card2_560x280", 100)):
    fig.savefig(f"{name}.png", dpi=dpi)
from PIL import Image
for name in ("card2_1120x560", "card2_560x280"):
    print(name, Image.open(f"{name}.png").size)
