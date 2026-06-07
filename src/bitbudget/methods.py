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


@method("float16", bits=16, axis="quantisation")
def float16(demb, qemb):
    """Half precision: the cheapest, most widely deployed compression. 2x smaller, essentially
    lossless, and natively supported by most vector stores."""
    dq = demb.astype(np.float16).astype(np.float32)
    return qemb @ dq.T, 2 * demb.shape[1]


@method("int8", bits=8)
def int8(demb, qemb):
    scale = np.abs(demb).max(0, keepdims=True) / 127.0 + 1e-9
    dq = np.round(demb / scale) * scale
    return qemb @ dq.T, demb.shape[1]


@method("int4", bits=4)
def int4(demb, qemb):
    """4-bit per-dimension scalar quantisation (15 signed levels). Half the footprint of int8,
    a common aggressive-but-not-binary operating point."""
    scale = np.abs(demb).max(0, keepdims=True) / 7.0 + 1e-9
    dq = np.clip(np.round(demb / scale), -7, 7) * scale
    return qemb @ dq.T, demb.shape[1] / 2.0


@method("binary", bits=1)
def binary(demb, qemb):
    return np.sign(qemb) @ np.sign(demb).T, demb.shape[1] / 8.0


@method("binary-mean", bits=1)
def binary_mean(demb, qemb):
    """Sign code thresholded at the per-coordinate mean rather than at zero -- the proper binary
    baseline, since a zero threshold assumes mean-centred data (LSH assumption A1). For strongly
    off-centre embeddings (e.g. e5) it recovers most of the quality a zero threshold wastes; for
    already-centred embeddings it is near-identical to `binary`."""
    mu = demb.mean(0, keepdims=True)
    return np.sign(qemb - mu) @ np.sign(demb - mu).T, demb.shape[1] / 8.0


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
def _kmeans(X, k, iters=10, seed=0, train_cap=16384):
    """Lloyd's k-means with a vectorised centroid update (bincount per sub-dimension, no Python
    loop over centroids) and a training subsample, as FAISS does, so PQ stays fast on large
    corpora and high-dimensional embeddings. Centroids are learned on the subsample; the returned
    assignment covers all of X."""
    rng = np.random.RandomState(seed)
    Xt = X if len(X) <= train_cap else X[rng.choice(len(X), train_cap, replace=False)]
    C = Xt[rng.choice(len(Xt), size=min(k, len(Xt)), replace=False)].copy()
    dim = Xt.shape[1]
    for _ in range(iters):
        a = ((Xt ** 2).sum(1, keepdims=True) - 2 * Xt @ C.T + (C ** 2).sum(1)).argmin(1)
        counts = np.bincount(a, minlength=len(C))
        sums = np.zeros((len(C), dim), dtype=np.float64)
        for c in range(dim):                              # loop over the small sub-dimension only
            sums[:, c] = np.bincount(a, weights=Xt[:, c], minlength=len(C))
        nz = counts > 0
        C[nz] = (sums[nz] / counts[nz, None]).astype(C.dtype)
    aall = ((X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1)).argmin(1)
    return C, aall


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
