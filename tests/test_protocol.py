"""Fast, dependency-light protocol tests (no downloads, no torch): verify the registry and
that every registered method returns a valid (scores, bytes) result under evaluate()."""
import numpy as np

from bitbudget import evaluate, list_methods, METHODS


def _toy(n_docs=64, n_q=8, dim=32, seed=0):
    rng = np.random.RandomState(seed)
    demb = rng.randn(n_docs, dim).astype("float32")
    demb /= np.linalg.norm(demb, axis=1, keepdims=True)
    qemb = rng.randn(n_q, dim).astype("float32")
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True)
    cids = [str(i) for i in range(n_docs)]
    qids = [str(i) for i in range(n_q)]
    exact = qemb @ demb.T
    qrels = {qids[i]: {cids[j]: 1 for j in np.argsort(-exact[i])[:3]} for i in range(n_q)}
    return demb, qemb, cids, qids, qrels


def test_registry_has_builtins():
    for m in ["float32", "int8", "binary", "binary+rerank", "rabitq", "pq", "matryoshka", "pca"]:
        assert m in METHODS, m


def test_methods_return_valid_results():
    demb, qemb, cids, qids, qrels = _toy()
    rows = evaluate(demb, qemb, cids, qids, qrels)
    assert len(rows) == len(list_methods())
    for r in rows:
        assert r["bytes"] > 0, r
        assert 0.0 <= r["ndcg"] <= 1.0, r
        assert 0.0 <= r["recall"] <= 1.0, r
    # rows are sorted by ascending footprint
    assert [r["bytes"] for r in rows] == sorted(r["bytes"] for r in rows)


def test_float32_is_lossless_baseline():
    demb, qemb, cids, qids, qrels = _toy()
    rows = {r["method"]: r for r in evaluate(demb, qemb, cids, qids, qrels)}
    assert rows["float32"]["pct_float"] == 100.0
    assert rows["float32"]["recall"] == 1.0          # exact == exact ground truth
    # a one-bit code plus re-ranking should recover essentially all of the float quality
    assert rows["binary+rerank"]["pct_float"] >= rows["binary"]["pct_float"]
