"""The BitBudget protocol: given document/query embeddings and judgements, score every
registered compression method on retrieval quality against the bytes it stores per vector."""
import numpy as np

from .methods import METHODS
from .metrics import ndcg_at_k, recall_at_k


def evaluate(demb, qemb, cids, qids, qrels, methods=None, recall_k=10, ndcg_k=10):
    """Run `methods` (default: all registered) and return a list of result dicts:
    {method, axis, bytes, ndcg, recall, pct_float}. nDCG uses the graded qrels; recall uses
    the exact float top-k as ground truth; pct_float is nDCG relative to the float32 baseline.
    """
    methods = methods or list(METHODS)
    exact = qemb @ demb.T                              # exact float scores = recall ground truth
    base = ndcg_at_k(exact, qids, cids, qrels, ndcg_k)
    rows = []
    for name in methods:
        fn = METHODS[name]
        scores, bpv = fn(demb, qemb)
        n = ndcg_at_k(scores, qids, cids, qrels, ndcg_k)
        r = recall_at_k(scores, exact, recall_k)
        rows.append(dict(method=name, axis=getattr(fn, "axis", "?"), bytes=float(bpv),
                         ndcg=round(n, 4), recall=round(r, 4),
                         pct_float=round(100 * n / base, 1) if base else float("nan")))
    rows.sort(key=lambda x: x["bytes"])
    return rows


def aggregate(per_corpus):
    """Mean +/- std across corpora for each method, given a {corpus: rows} mapping."""
    by = {}
    for corpus, rows in per_corpus.items():
        for r in rows:
            by.setdefault(r["method"], []).append(r)
    out = []
    for name, rs in by.items():
        nd = np.array([r["ndcg"] for r in rs]);  pc = np.array([r["pct_float"] for r in rs])
        out.append(dict(method=name, axis=rs[0]["axis"], bytes=rs[0]["bytes"],
                        ndcg=round(float(nd.mean()), 3), ndcg_std=round(float(nd.std()), 3),
                        pct_float=round(float(pc.mean()), 1), pct_float_std=round(float(pc.std()), 1),
                        n_corpora=len(rs)))
    out.sort(key=lambda x: x["bytes"])
    return out
