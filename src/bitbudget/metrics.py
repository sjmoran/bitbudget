"""Retrieval metrics. nDCG@k uses graded BEIR judgements; recall@k uses the exact
floating-point neighbours as ground truth (the ANN-Benchmarks convention)."""
import numpy as np


def _topk(scores_row, k):
    k = min(k, scores_row.shape[0])
    idx = np.argpartition(-scores_row, k - 1)[:k]
    return idx[np.argsort(-scores_row[idx])]


def ndcg_at_k(scores, qids, cids, qrels, k=10):
    """Mean nDCG@k over queries with graded relevance judgements `qrels`."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = []
    cids = [str(c) for c in cids]
    for i, q in enumerate(qids):
        rel = qrels.get(str(q), {})
        if not rel:
            continue
        top = _topk(scores[i], k)
        gains = np.array([rel.get(cids[j], 0) for j in top], dtype=float)
        dcg = np.sum((2 ** gains - 1) * disc[:len(gains)])
        ideal = np.sort(np.array(list(rel.values()), dtype=float))[::-1][:k]
        idcg = np.sum((2 ** ideal - 1) * disc[:len(ideal)])
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(out)) if out else float("nan")


def recall_at_k(scores, exact_scores, k=10):
    """k-recall@k against the exact floating-point top-k (ground truth from `exact_scores`)."""
    out = []
    for i in range(scores.shape[0]):
        gt = set(_topk(exact_scores[i], k).tolist())
        got = set(_topk(scores[i], k).tolist())
        out.append(len(gt & got) / float(k))
    return float(np.mean(out))
