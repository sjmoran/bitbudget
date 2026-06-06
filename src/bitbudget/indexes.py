"""The organisation axis: build an index over document vectors and measure recall@k, query
throughput (QPS) and the bytes it stores per vector.

This is the axis the compression leaderboard does not cover. A graph index such as HNSW does not
shrink the footprint, it *adds* bytes (the vectors plus a graph) and buys throughput; a compact
code such as the bit-trie shrinks it. Reporting recall, QPS and bytes together makes the
trade-off explicit, in the spirit of ann-benchmarks.

Register an index with ``@index``. The faiss-backed indexes (flat/hnsw/ivfpq) import faiss lazily
and are skipped with a message if faiss is not installed; the bit-trie is numpy-only. Vectors are
L2-normalised on ingest so inner-product and L2 rankings agree. Run in its own process (no torch)
to avoid the faiss/OpenMP clash (see README).
"""
import time
import numpy as np

INDEXES = {}


def index(name, axis="organisation"):
    """Register an index. An index is ``fn(xb, xq, k) -> (topk_indices, bytes_per_vec, qps)``."""
    def deco(fn):
        fn.bitbudget_name = name
        fn.axis = axis
        INDEXES[name] = fn
        return fn
    return deco


def list_indexes():
    return sorted(INDEXES)


def _time(search, repeats=3):
    best, I = float("inf"), None
    for _ in range(repeats):
        t = time.time(); I = search(); best = min(best, time.time() - t)
    return I, best


# ----------------------------------------------------------------- faiss-backed (organisation)
@index("flat")
def flat(xb, xq, k):
    import faiss
    D = xb.shape[1]
    ix = faiss.IndexFlatIP(D); ix.add(xb)
    I, t = _time(lambda: ix.search(xq, k)[1])
    return I, 4 * D, len(xq) / t                       # full float vectors, exact


@index("hnsw")
def hnsw(xb, xq, k, M=24, efc=200, efs=128):
    import faiss
    D = xb.shape[1]
    ix = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
    ix.hnsw.efConstruction = efc; ix.add(xb); ix.hnsw.efSearch = efs
    I, t = _time(lambda: ix.search(xq, k)[1])
    return I, 4 * D + M * 2 * 4, len(xq) / t           # vectors + graph edges (both levels approx)


@index("ivfpq")
def ivfpq(xb, xq, k, nprobe=64):
    import faiss
    n, D = xb.shape
    m = D // 2
    while D % m:
        m -= 1
    nlist = max(1, min(8192, n // 100))
    ix = faiss.IndexIVFPQ(faiss.IndexFlatIP(D), D, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    samp = xb[np.random.RandomState(0).choice(n, min(200000, n), replace=False)]
    ix.train(samp); ix.add(xb); ix.nprobe = min(nprobe, nlist)
    I, t = _time(lambda: ix.search(xq, k)[1])
    return I, float(m), len(xq) / t                    # PQ code bytes


# ----------------------------------------------------------------- numpy bit-trie (research entry)
@index("bittrie", axis="quantisation")
def bittrie(xb, xq, k, nbits=64, beam=128, depth=24):
    from .bittrie import BitTrieIndex
    ix = BitTrieIndex(n_bits=nbits, seed=0).fit(xb)
    I, t = _time(lambda: np.array([ix.query(q, topk=k, beam=beam, depth=depth, rerank="float")
                                   for q in xq]), repeats=1)
    return I, nbits / 8.0, len(xq) / t                 # compact routing code (float kept cold for re-rank)


# ----------------------------------------------------------------- harness
def _exact(xb, xq, k, chunk=200000):
    """Exact top-k by inner product, chunked over documents to bound memory."""
    nq = len(xq)
    best_s = np.full((nq, k), -1e30, np.float32)
    best_i = np.zeros((nq, k), np.int64)
    for s in range(0, len(xb), chunk):
        e = min(s + chunk, len(xb))
        sc = xq @ xb[s:e].T
        cs = np.concatenate([best_s, sc], 1)
        ci = np.concatenate([best_i, np.broadcast_to(np.arange(s, e), (nq, e - s))], 1)
        part = np.argpartition(-cs, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(cs, part, 1)
        best_i = np.take_along_axis(ci, part, 1)
    order = np.argsort(-best_s, 1)
    return np.take_along_axis(best_i, order, 1)


def bench_indexes(xb, xq, k=10, gt=None, which=None, verbose=True):
    """Build each index and report recall@k, QPS and bytes/vec. Returns rows sorted by footprint.
    Vectors are L2-normalised so the inner-product ground truth is metric-consistent across indexes.
    """
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    xb /= (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-9)
    xq /= (np.linalg.norm(xq, axis=1, keepdims=True) + 1e-9)
    if gt is None:
        gt = _exact(xb, xq, k)
    rows = []
    for name in (which or list_indexes()):
        fn = INDEXES[name]
        try:
            I, bpv, qps = fn(xb, xq, k)
        except ImportError:
            if verbose:
                print(f"  [skip {name}: install bitbudget[faiss]]")
            continue
        rec = float(np.mean([len(set(I[i][:k].tolist()) & set(gt[i][:k].tolist())) / k
                             for i in range(len(xq))]))
        rows.append(dict(method=name, axis=getattr(fn, "axis", "organisation"),
                         bytes=float(bpv), recall=round(rec, 4), qps=round(float(qps), 1)))
        if verbose:
            print(f"  {name:14s}{bpv:8.0f}B  recall@{k}={rec:.3f}  {qps:9.0f} qps")
    rows.sort(key=lambda r: r["bytes"])
    return rows
