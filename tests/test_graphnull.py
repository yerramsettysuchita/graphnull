"""Tests for graphnull.py on small synthetic graphs. No network, no competition data."""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graphnull as gn  # noqa: E402


def toy_pairs(n_nodes=40, n_edges=120, n_groups=4, seed=0):
    """Random simple graph whose node ids look like module.function, mostly within-module edges."""
    rng = random.Random(seed)
    nodes = [f"pkg.mod{i % n_groups}.f{i}" for i in range(n_nodes)]
    pairs = set()
    while len(pairs) < n_edges:
        u, v = rng.sample(nodes, 2)
        if group_of(u) != group_of(v) and rng.random() < 0.7:
            continue
        pairs.add((u, v) if u < v else (v, u))
    return nodes, pairs


def group_of(node):
    return node.rsplit(".", 1)[0]


def test_rewire_degree_keeps_degrees_and_edge_count_and_changes_edges():
    _, pairs = toy_pairs()
    null, done = gn.rewire_degree(pairs, seed=1)
    chk = gn.check_null(pairs, null)
    assert chk["degree_preserved"]
    assert chk["same_edge_count"]
    assert gn.degrees(null) == gn.degrees(pairs)
    assert chk["edges_changed"] > 0
    assert null != pairs
    assert 0 < done <= 1


def test_rewire_stratified_keeps_degrees_and_group_profile():
    _, pairs = toy_pairs()
    null, _ = gn.rewire_stratified(pairs, group_of, seed=1)
    chk = gn.check_null(pairs, null, group_of)
    assert chk["degree_preserved"]
    assert chk["same_edge_count"]
    assert chk["group_profile_preserved"] is True
    assert chk["edges_changed"] > 0


def test_random_same_size_keeps_edge_count_and_node_set():
    nodes, pairs = toy_pairs()
    null = gn.random_same_size(nodes, len(pairs), seed=1)
    assert len(null) == len(pairs)
    endpoints = {x for e in null for x in e}
    assert endpoints <= set(nodes)
    assert all(u != v for u, v in null)


def test_same_seed_same_result_and_different_seed_different_result():
    nodes, pairs = toy_pairs()
    assert gn.rewire_degree(pairs, 5) == gn.rewire_degree(pairs, 5)
    assert gn.rewire_degree(pairs, 5)[0] != gn.rewire_degree(pairs, 6)[0]
    assert gn.rewire_stratified(pairs, group_of, 5) == gn.rewire_stratified(pairs, group_of, 5)
    assert gn.rewire_stratified(pairs, group_of, 5)[0] != gn.rewire_stratified(pairs, group_of, 6)[0]
    assert gn.random_same_size(nodes, 60, 5) == gn.random_same_size(nodes, 60, 5)
    assert gn.random_same_size(nodes, 60, 5) != gn.random_same_size(nodes, 60, 6)
    keys = [f"n{i}" for i in range(30)]
    m = np.arange(30 * 4, dtype=np.float32).reshape(30, 4)
    assert np.array_equal(gn.permute_embeddings(keys, m, 5), gn.permute_embeddings(keys, m, 5))
    assert not np.array_equal(gn.permute_embeddings(keys, m, 5), gn.permute_embeddings(keys, m, 6))
    assert gn.name_seed("a") == gn.name_seed("a")
    assert gn.name_seed("a") != gn.name_seed("b")


def test_permute_embeddings_keeps_the_set_of_vectors():
    keys = [f"n{i}" for i in range(30)]
    m = np.random.default_rng(0).normal(size=(30, 8)).astype(np.float32)
    p = gn.permute_embeddings(keys, m, seed=3)
    assert p.shape == m.shape
    assert p.dtype == m.dtype
    assert sorted(map(tuple, p.tolist())) == sorted(map(tuple, m.tolist()))
    assert not np.array_equal(p, m)


def write_inputs(root):
    gdir, edir = os.path.join(root, "graphs"), os.path.join(root, "embeddings")
    os.makedirs(gdir)
    os.makedirs(edir)
    for k, name in enumerate(["repo_a", "repo_b"]):
        nodes, pairs = toy_pairs(seed=k)
        edges = [{"source": u, "target": v, "type": "calls"} for u, v in sorted(pairs)]
        with open(os.path.join(gdir, name + ".json"), "w") as f:
            json.dump({"nodes": [{"id": n} for n in nodes], "edges": edges}, f)
        rng = np.random.default_rng(k)
        np.savez(os.path.join(edir, name + ".npz"),
                 **{n: rng.normal(size=8).astype(np.float32) for n in nodes})
    return gdir, edir


def test_make_worlds_writes_all_worlds_and_is_deterministic(tmp_path):
    gdir, edir = write_inputs(str(tmp_path / "in"))
    out1, out2 = str(tmp_path / "out1"), str(tmp_path / "out2")
    m1, h1, rep = gn.make_worlds(gdir, edir, out1, stratified_group=group_of, verbose=False)
    for w in ("W1", "W2", "W3", "W2S"):
        for name in ("repo_a", "repo_b"):
            assert os.path.exists(os.path.join(out1, w, "graphs", name + ".json")), (w, name)
            assert os.path.exists(os.path.join(out1, w, "embeddings", name + ".npz")), (w, name)
    assert os.path.exists(os.path.join(out1, "manifest.json"))
    assert isinstance(h1, str) and len(h1) == 16
    assert all(rep["degree_preserved"])
    assert all(rep["stratified_profile_preserved"])
    assert all(rep["embeddings_same_keys"])
    m2, h2, _ = gn.make_worlds(gdir, edir, out2, stratified_group=group_of, verbose=False)
    assert h2 == h1
    assert m2 == m1


def test_copy_fallback_gives_the_same_worlds_as_symlinks(tmp_path, monkeypatch):
    """1.0.0 linked files with os.symlink; 1.0.1 copies when that fails. Worlds and hash must not change."""
    import shutil
    gdir, edir = write_inputs(str(tmp_path / "in"))

    def symlink_as_copy(src, dst):  # what a working symlink gives a reader: the source bytes
        shutil.copyfile(src, dst)

    def symlink_refused(src, dst):
        raise OSError("symbolic links are not allowed here")

    outs, hashes = {}, {}
    for label, fake in (("linked", symlink_as_copy), ("refused", symlink_refused)):
        monkeypatch.setattr(gn.os, "symlink", fake)
        outs[label] = str(tmp_path / label)
        _, hashes[label], _ = gn.make_worlds(gdir, edir, outs[label], stratified_group=group_of, verbose=False)
        monkeypatch.undo()
    assert hashes["linked"] == hashes["refused"]
    for w in ("W1", "W2", "W3"):
        for name in ("repo_a", "repo_b"):
            for sub, ext in (("graphs", ".json"), ("embeddings", ".npz")):
                rel = os.path.join(w, sub, name + ext)
                with open(os.path.join(outs["linked"], rel), "rb") as a, open(os.path.join(outs["refused"], rel), "rb") as b:
                    assert a.read() == b.read(), rel
