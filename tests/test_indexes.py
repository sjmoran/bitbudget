"""Index-benchmark tests. The bit-trie path is numpy-only, so these run in CI without faiss;
the faiss-backed indexes (flat/hnsw/ivfpq) are exercised only if faiss is importable."""
import numpy as np

from bitbudget import INDEXES, list_indexes, bench_indexes


def _clustered(n=4000, d=32, nq=100, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(max(2, n // 200), d).astype("float32") * 4.0
    xb = centers[rng.randint(0, len(centers), n)] + rng.randn(n, d).astype("float32")
    xq = xb[rng.choice(n, nq, replace=False)] + 0.1 * rng.randn(nq, d).astype("float32")
    return xb, xq


def test_registry_has_indexes():
    for name in ["flat", "hnsw", "ivfpq", "bittrie"]:
        assert name in INDEXES, name
    assert INDEXES["hnsw"].axis == "organisation"
    assert INDEXES["bittrie"].axis == "quantisation"


def test_bittrie_bench_runs_without_faiss():
    xb, xq = _clustered()
    rows = bench_indexes(xb, xq, k=10, which=["bittrie"], verbose=False)
    assert len(rows) == 1
    r = rows[0]
    assert r["method"] == "bittrie"
    assert r["bytes"] == 8.0                      # 64-bit code
    assert 0.0 <= r["recall"] <= 1.0
    assert r["qps"] > 0
    assert r["recall"] > 0.5                       # clustered data is easy; should be well above chance


def test_query_batch_matches_python():
    """The C kernel (when compiled) must return the same top-k as the Python path; if no compiler
    is available, query_batch falls back to Python and trivially matches."""
    from bitbudget.bittrie import BitTrieIndex
    xb, xq = _clustered(n=6000, d=64, nq=120)
    ix = BitTrieIndex(n_bits=64, seed=0).fit(xb)
    py = np.array([ix.query(q, topk=10, beam=128, depth=24, rerank="float") for q in xq])
    batch = ix.query_batch(xq, topk=10, beam=128, depth=24)
    overlap = np.mean([len(set(py[i].tolist()) & set(batch[i].tolist())) / 10 for i in range(len(xq))])
    assert overlap > 0.99, overlap


def test_bench_is_sorted_by_footprint():
    xb, xq = _clustered()
    rows = bench_indexes(xb, xq, k=10, verbose=False)   # all available (faiss skipped if absent)
    assert [r["bytes"] for r in rows] == sorted(r["bytes"] for r in rows)
    for r in rows:
        assert r["bytes"] > 0 and r["qps"] > 0 and 0.0 <= r["recall"] <= 1.0
