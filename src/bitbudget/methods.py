"""Compression methods. A method maps float document/query embeddings to a
query-by-document similarity matrix under some compression, and reports the bytes it stores
per document vector. Register your own with the ``@method`` decorator.

    @method("my-method", bits=2)
    def my_method(demb, qemb):
        ...
        return scores, bytes_per_vec        # scores: (n_queries, n_docs); bytes: float

Everything here is numpy-only so that evaluation runs in the same process as torch embedding
without the faiss/OpenMP clash (see README). faiss-backed methods live in methods_faiss.py and
run in a separate evaluation process.
"""
import numpy as np

METHODS = {}


def method(name, bits=None, axis="quantisation"):
    """Register a compression method. `bits` and `axis` are metadata for the leaderboard."""
    def deco(fn):
        fn.bitbudget_name = name
        fn.bits = bits
        fn.axis = axis
        METHODS[name] = fn
        return fn
    return deco


def list_methods():
    return sorted(METHODS)


# ---------------------------------------------------------------- baselines / quantisation
@method("float32", bits=32, axis="none")
def float32(demb, qemb):
    return qemb @ demb.T, 4 * demb.shape[1]


@method("int8", bits=8)
def int8(demb, qemb):
    scale = np.abs(demb).max(0, keepdims=True) / 127.0 + 1e-9
    dq = np.round(demb / scale) * scale
    return qemb @ dq.T, demb.shape[1]


@method("binary", bits=1)
def binary(demb, qemb):
    return np.sign(qemb) @ np.sign(demb).T, demb.shape[1] / 8.0


@method("binary+rerank", bits=1)
def binary_rerank(demb, qemb, depth=100):
    bq, bd = np.sign(qemb), np.sign(demb)
    cand = np.argsort(-(bq @ bd.T), axis=1)[:, :depth]
    scores = np.full((qemb.shape[0], demb.shape[0]), -1e9, dtype=np.float32)
    for i in range(qemb.shape[0]):
        scores[i, cand[i]] = qemb[i] @ demb[cand[i]].T
    return scores, demb.shape[1] / 8.0          # compact code stored; full vectors fetched on re-rank


@method("rabitq", bits=1)
def rabitq(demb, qemb, seed=0):
    D = demb.shape[1]
    R = np.linalg.qr(np.random.RandomState(seed).randn(D, D))[0].astype(np.float32)
    return (qemb @ R) @ np.sign(demb @ R).T, D / 8.0


# ---------------------------------------------------------------- product quantisation (numpy)
def _kmeans(X, k, iters=10, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
    for _ in range(iters):
        d = ((X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1))
        a = d.argmin(1)
        for j in range(len(C)):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return C, a


@method("pq", bits=8, axis="quantisation")
def pq(demb, qemb, m=None, ksub=256):
    """Product quantisation: m sub-quantisers, 256 centroids each (m bytes/vector). Asymmetric
    scoring against reconstructed documents."""
    D = demb.shape[1]
    m = m or max(1, D // 8)
    assert D % m == 0, f"PQ needs D ({D}) divisible by m ({m})"
    sub = D // m
    recon = np.empty_like(demb)
    for s in range(m):
        sl = slice(s * sub, (s + 1) * sub)
        C, a = _kmeans(demb[:, sl], ksub, seed=s)
        recon[:, sl] = C[a]
    return qemb @ recon.T, float(m)


# ---------------------------------------------------------------- projection axis (dimensions)
@method("matryoshka", bits=32, axis="projection")
def matryoshka(demb, qemb, dim=None):
    """Prefix-truncation. Fair only for Matryoshka-trained embedders; on others it is naive."""
    dim = dim or demb.shape[1] // 4
    return qemb[:, :dim] @ demb[:, :dim].T, 4 * dim


@method("pca", bits=32, axis="projection")
def pca(demb, qemb, dim=None):
    dim = dim or demb.shape[1] // 4
    mean = demb.mean(0)
    dc = demb - mean
    _, V = np.linalg.eigh(dc.T @ dc)
    W = V[:, ::-1][:, :dim]
    return ((qemb - mean) @ W) @ (dc @ W).T, 4 * dim
