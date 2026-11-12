# Copyright 2026 Yerramsetty Sai Venkata Suchita, Manjunath Mailey, Kanuma Jagan Mohan.
# Licensed under the Apache License, Version 2.0.
"""GraphNull: interface-matched null worlds for code-graph tools.

A null world keeps everything an agent can see about a graph tool (tool names,
node set, output format, every node's degree) and scrambles only the content
that a structural claim depends on. If an agent does as well in the null world
as in the real one, the gain cannot be credited to the structure.

Null ladder, from weakest to strongest:
  random      same node set and edge count, edges placed uniformly (degrees change)
  degree      degree-preserving double-edge swaps (every node keeps its degree)
  stratified  degree-preserving swaps that also keep, for every node, how many of
              its neighbours sit in each group (for example each source file)
  permuted    real graph, embedding vectors permuted across nodes

Only the standard library and numpy are needed. Every function is
deterministic given its seed. The harness file format is a JSON graph with
"nodes" and "edges" (source, target, type) and an .npz embedding file keyed by
node id, as in the Gemma 4 Developer Agent competition.
"""
import collections
import glob
import hashlib
import json
import os
import random
import shutil

import numpy as np

__version__ = "1.0.1"
DEFAULT_SEED = 20261108
SWAPS_PER_EDGE = 10


# ---------------------------------------------------------------- seeds and io
def name_seed(name, seed=DEFAULT_SEED):
    """Stable per-object seed derived from a global seed and a name."""
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:12], 16)


def normalize(edges):
    """Edge dicts or (u, v) tuples to a set of undirected pairs and a set of self loops."""
    pairs, loops = set(), set()
    for e in edges:
        u, v = (e["source"], e["target"]) if isinstance(e, dict) else e
        if u == v:
            loops.add(u)
        else:
            pairs.add((u, v) if u < v else (v, u))
    return pairs, loops


def degrees(pairs):
    d = collections.Counter()
    for u, v in pairs:
        d[u] += 1
        d[v] += 1
    return d


def adjacency(pairs):
    a = collections.defaultdict(set)
    for u, v in pairs:
        a[u].add(v)
        a[v].add(u)
    return a


def emit(graph, pairs, loops, edge_type="calls"):
    """Write pairs back in the harness format: both directions, one edge type."""
    edges = []
    for u, v in sorted(pairs):
        edges.append({"source": u, "target": v, "type": edge_type})
        edges.append({"source": v, "target": u, "type": edge_type})
    for u in sorted(loops):
        edges.append({"source": u, "target": u, "type": edge_type})
    out = {k: v for k, v in graph.items() if k not in ("edges", "links")}
    out["edges"] = edges
    return out


def write_json(obj, path):
    blob = json.dumps(obj, sort_keys=True).encode()
    with open(path, "wb") as f:
        f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _link_or_copy(src, dst):
    """Symlink dst to src; copy the file where symlinks are not allowed (for example Windows without Developer Mode)."""
    if os.path.lexists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copyfile(src, dst)


# ---------------------------------------------------------------- null models
def rewire_degree(pairs, seed, swaps_per_edge=SWAPS_PER_EDGE, accept=None):
    """Degree-preserving double-edge swaps.

    Picks two edges (a, b) and (c, d) and replaces them with (a, d) and (c, b)
    unless that creates a self loop or a duplicate edge. `accept(a, b, c, d)`
    can veto a swap (used by the stratified null). Returns (new pair set, share of the
    requested swaps that were completed).
    """
    rng = random.Random(seed)
    E = sorted(pairs)
    S = set(E)
    m = len(E)
    done, tries = 0, 0
    while done < swaps_per_edge * m and tries < swaps_per_edge * m * 50:
        tries += 1
        i = int(rng.random() * m)
        j = int(rng.random() * m)
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        if accept is not None and not accept(a, b, c, d):
            continue
        e1 = (a, d) if a < d else (d, a)
        e2 = (c, b) if c < b else (b, c)
        if e1 in S or e2 in S:
            continue
        S.discard(E[i])
        S.discard(E[j])
        S.add(e1)
        S.add(e2)
        E[i], E[j] = e1, e2
        done += 1
    return set(E), done / max(swaps_per_edge * m, 1)


def rewire_stratified(pairs, group_of, seed, swaps_per_edge=SWAPS_PER_EDGE):
    """Degree-preserving swaps restricted to partners in the same group.

    A swap (a, b), (c, d) -> (a, d), (c, b) is allowed only when b and d are in
    the same group and a and c are in the same group. Then each of the four
    endpoints gives up one neighbour and receives another from the same group,
    so every node keeps both its degree and its count of neighbours per group. With group = source file this
    keeps file locality and scrambles which function in the file is called.
    """
    return rewire_degree(pairs, seed, swaps_per_edge,
                         accept=lambda a, b, c, d: group_of(b) == group_of(d) and group_of(a) == group_of(c))


def random_same_size(nodes, n_edges, seed):
    """Uniform random simple graph on the same nodes with the same edge count."""
    rng = random.Random(seed)
    nodes = sorted(nodes)
    out = set()
    if len(nodes) < 2:
        return out
    cap = len(nodes) * (len(nodes) - 1) // 2
    n_edges = min(n_edges, cap)
    while len(out) < n_edges:
        u, v = rng.sample(nodes, 2)
        out.add((u, v) if u < v else (v, u))
    return out


def permute_embeddings(keys, matrix, seed):
    """Same keys, same vectors, assigned to different nodes."""
    perm = np.random.default_rng(seed).permutation(len(keys))
    return matrix[perm]


# ---------------------------------------------------------------- checks
def group_profile(pairs, group_of):
    """For every node, a Counter of neighbour groups."""
    prof = collections.defaultdict(collections.Counter)
    for u, v in pairs:
        prof[u][group_of(v)] += 1
        prof[v][group_of(u)] += 1
    return prof


def check_null(real, null, group_of=None):
    """Invariants of a null graph against the real one."""
    out = {
        "same_edge_count": len(real) == len(null),
        "degree_preserved": degrees(real) == degrees(null),
        "edges_changed": 1 - len(real & null) / max(len(real), 1),
    }
    if group_of is not None:
        out["group_profile_preserved"] = group_profile(real, group_of) == group_profile(null, group_of)
    return out


# ---------------------------------------------------------------- harness worlds
def make_worlds(graph_dir, emb_dir, out_dir, seed=DEFAULT_SEED, swaps_per_edge=SWAPS_PER_EDGE,
                stratified_group=None, verbose=True):
    """Build harness-compatible worlds from a directory of graphs and embeddings.

    W1 real graph (normalized), W2 degree-preserving rewire, W3 real graph with
    permuted embeddings. If `stratified_group` is a function node_id -> group,
    W2S (degree and group preserving) is also written. Graph and embedding files
    keep their names, so a harness pointed at out_dir/W2 cannot tell the worlds
    apart by interface. Returns (manifest dict, manifest sha256 prefix, report).
    """
    gfiles = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(graph_dir, "*.json"))}
    efiles = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(emb_dir, "*.npz"))}
    worlds = ["W1", "W2", "W3"] + (["W2S"] if stratified_group else [])
    for w in worlds:
        for sub in ("graphs", "embeddings"):
            os.makedirs(os.path.join(out_dir, w, sub), exist_ok=True)
    manifest = {"seed": seed, "files": {}}
    report = collections.defaultdict(list)
    for k, name in enumerate(sorted(gfiles)):
        with open(gfiles[name]) as f:
            G = json.load(f)
        pairs, loops = normalize(G["edges"])
        s = name_seed(name, seed)
        rpairs, frac = rewire_degree(pairs, s, swaps_per_edge)
        chk = check_null(pairs, rpairs)
        report["swaps_completed"].append(frac)
        report["degree_preserved"].append(chk["degree_preserved"])
        report["edges_changed"].append(chk["edges_changed"])
        g1 = os.path.join(out_dir, "W1", "graphs", name + ".json")
        g2 = os.path.join(out_dir, "W2", "graphs", name + ".json")
        g3 = os.path.join(out_dir, "W3", "graphs", name + ".json")
        h1 = write_json(emit(G, pairs, loops), g1)
        h2 = write_json(emit(G, rpairs, loops), g2)
        _link_or_copy(g1, g3)
        entry = {"W1_graph": h1, "W2_graph": h2}
        if stratified_group:
            spairs, _ = rewire_stratified(pairs, stratified_group, s, swaps_per_edge)
            sc = check_null(pairs, spairs, stratified_group)
            report["stratified_profile_preserved"].append(sc["group_profile_preserved"] and sc["degree_preserved"])
            report["stratified_edges_changed"].append(sc["edges_changed"])
            gs = os.path.join(out_dir, "W2S", "graphs", name + ".json")
            entry["W2S_graph"] = write_json(emit(G, spairs, loops), gs)
        if name in efiles:
            src = efiles[name]
            for w in ["W1", "W2"] + (["W2S"] if stratified_group else []):
                dst = os.path.join(out_dir, w, "embeddings", name + ".npz")
                _link_or_copy(src, dst)
            E = np.load(src)
            keys = list(E.keys())
            M = np.stack([E[x] for x in keys])
            Mp = permute_embeddings(keys, M, s)
            dst3 = os.path.join(out_dir, "W3", "embeddings", name + ".npz")
            np.savez(dst3, **{x: Mp[i] for i, x in enumerate(keys)})
            E3 = np.load(dst3)
            report["embeddings_same_keys"].append(sorted(E3.keys()) == sorted(keys) and E3[keys[0]].dtype == M.dtype)
            entry["emb"] = file_hash(src)
            entry["W3_emb"] = file_hash(dst3)
        if stratified_group:
            manifest["files"][name] = entry
        else:
            manifest["files"][name] = {key: entry[key] for key in ("W1_graph", "W2_graph", "emb", "W3_emb") if key in entry}
        if verbose and k % 25 == 0:
            print(f"  graph {k + 1}/{len(gfiles)}")
    blob = json.dumps(manifest, sort_keys=True).encode()
    with open(os.path.join(out_dir, "manifest.json"), "wb") as f:
        f.write(blob)
    return manifest, hashlib.sha256(blob).hexdigest()[:16], dict(report)


def summarize(report):
    """One line per invariant: count of passes and min / median of shares."""
    lines = []
    for key, vals in report.items():
        if vals and isinstance(vals[0], bool):
            lines.append(f"{key}: {sum(vals)} / {len(vals)}")
        elif vals:
            lines.append(f"{key}: min {min(vals):.3f} | median {float(np.median(vals)):.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build GraphNull worlds for a harness graph directory.")
    ap.add_argument("graph_dir")
    ap.add_argument("emb_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--swaps-per-edge", type=int, default=SWAPS_PER_EDGE)
    ap.add_argument("--stratify-by-file", action="store_true",
                    help="also write W2S, grouping nodes by the module part of their id")
    a = ap.parse_args()
    grp = (lambda n: n.rsplit(".", 1)[0]) if a.stratify_by_file else None
    _, h, rep = make_worlds(a.graph_dir, a.emb_dir, a.out_dir, a.seed, a.swaps_per_edge, grp)
    print(summarize(rep))
    print("manifest sha256:", h)
