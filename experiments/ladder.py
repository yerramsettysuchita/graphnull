# Copyright 2026 Yerramsetty Sai Venkata Suchita, Manjunath Mailey, Kanuma Jagan Mohan.
# Licensed under the Apache License, Version 2.0.
# GraphNull null ladder on three public repositories (click, flask, attrs).
# CPU only, internet on (clones the repositories). Reads no competition data.
# For each recent commit that edits 2 to 10 functions, asks how often the
# neighbour list of one edited function contains another edited function,
# in the real call graph and in three null graphs of increasing strength.
import collections, hashlib, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
sys.path.insert(0, os.path.dirname(os.path.abspath(HERE)))
import graphnull as gn

src = open(os.path.join(HERE, "heldout_core.py")).read()
exec(src)  # REPOS, SEED, git, is_source, module_of, index_file, build_graph, changed_old_lines, source_at, gold_nodes, recall, boot_ci, WORK

N_NULLS = int(os.environ.get("N_NULLS", "20"))
LEVELS = ["random", "degree", "stratified"]
t0 = time.time()
rows, inv = [], collections.defaultdict(list)
for name, url, pin in REPOS:
    repo = f"{WORK}/{name}"
    if not os.path.isdir(repo):
        subprocess.run(["git", "clone", "-q", url, repo], check=True)
    commits = git(repo, "log", "--no-merges", "--format=%H %P", "-n", str(N_COMMITS), pin, "--", "*.py").split("\n")
    kept = 0
    for line in commits:
        parts = line.split()
        if len(parts) != 2:
            continue
        commit, parent = parts
        touched = {p: l for p, l in changed_old_lines(repo, parent, commit).items() if is_source(p)}
        if not touched or len(touched) > 8:
            continue
        ids, pairs, spans = build_graph(source_at(repo, parent))
        gold = gold_nodes(spans, touched)
        if not (2 <= len(gold) <= 10) or len(pairs) < 50:
            continue
        grp = {nid: module_of(path) for path, sp in spans.items() for _, _, nid in sp}
        group_of = lambda n: grp.get(n, "?")
        adj1 = gn.adjacency(pairs)
        row = {"repo": name, "commit": commit[:12], "nodes": len(ids), "edges": len(pairs), "gold": len(gold),
               "same_file_share": float(np.mean([group_of(u) == group_of(v) for u, v in pairs]))}
        for h in (1, 2):
            row[f"real_{h}"] = recall(adj1, gold, h)
        acc = {(lv, h): [] for lv in LEVELS for h in (1, 2)}
        for k in range(N_NULLS):
            s = gn.name_seed(f"{name}:{commit}:{k}", SEED)
            nulls = {
                "random": gn.random_same_size(ids, len(pairs), s),
                "degree": gn.rewire_degree(pairs, s)[0],
                "stratified": gn.rewire_stratified(pairs, group_of, s)[0],
            }
            for lv, npairs in nulls.items():
                c = gn.check_null(pairs, npairs, group_of)
                for key, v in c.items():
                    inv[(lv, key)].append(v)
                a2 = gn.adjacency(npairs)
                for h in (1, 2):
                    acc[(lv, h)].append(recall(a2, gold, h))
        for (lv, h), v in acc.items():
            row[f"{lv}_{h}"] = float(np.mean(v))
        rows.append(row)
        kept += 1
    print(f"{name}: kept {kept} tasks | {round(time.time() - t0)} s", flush=True)

def fmt(t):
    return f"{100 * t[0]:5.1f} [{100 * t[1]:5.1f}, {100 * t[2]:5.1f}]"

print("\n=== NULL INVARIANTS (all repositories, all null graphs) ===")
for lv in LEVELS:
    d = {k[1]: v for k, v in inv.items() if k[0] == lv}
    print(f"{lv:10s} same edge count {sum(d['same_edge_count'])}/{len(d['same_edge_count'])}"
          f" | degree kept {sum(d['degree_preserved'])}/{len(d['degree_preserved'])}"
          f" | file profile kept {sum(d['group_profile_preserved'])}/{len(d['group_profile_preserved'])}"
          f" | edges changed median {np.median(d['edges_changed']):.3f} min {min(d['edges_changed']):.3f}")

print(f"\n=== NULL LADDER: co-edit recall (%), {len(rows)} commits, task bootstrap 95% CI ===")
print("share of real call edges inside one file, median:", round(float(np.median([r['same_file_share'] for r in rows])), 3))
for h in (1, 2):
    print(f"\n{h}-hop")
    for lv in ["real"] + LEVELS[::-1]:
        print(f"  {lv:10s} {fmt(boot_ci([r[f'{lv}_{h}'] for r in rows]))}")
    for lv in LEVELS[::-1]:
        print(f"  real - {lv:10s} {fmt(boot_ci([r[f'real_{h}'] - r[f'{lv}_{h}'] for r in rows]))}")
    print(f"  stratified - degree  {fmt(boot_ci([r[f'stratified_{h}'] - r[f'degree_{h}'] for r in rows]))}")
    print(f"  degree - random      {fmt(boot_ci([r[f'degree_{h}'] - r[f'random_{h}'] for r in rows]))}")
for name, _, _ in REPOS:
    rs = [r for r in rows if r["repo"] == name]
    print(f"{name:6s} n={len(rs):3d} 1-hop " + "  ".join(f"{lv}={100*np.mean([r[f'{lv}_1'] for r in rs]):.1f}" for lv in ["real"] + LEVELS[::-1]))

blob = json.dumps({"seed": SEED, "n_nulls": N_NULLS, "repos": REPOS, "rows": rows}, sort_keys=True).encode()
out = "/kaggle/working/ladder_results.json" if os.path.isdir("/kaggle/working") else "ladder_results.json"
open(out, "wb").write(blob)
print("\nresults sha256:", hashlib.sha256(blob).hexdigest()[:16])
print("graphnull version:", gn.__version__, "| total seconds:", round(time.time() - t0))
print("LADDER DONE")
