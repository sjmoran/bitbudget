"""Bit-trie index: a van Emde Boas / PATRICIA-style radix trie over compact codes.

The sorted array of packed sign-codes *is* the trie: a node is a contiguous range of codes
sharing a bit prefix. Search is coarse-to-fine beam descent (the most discriminative bit is the
shallowest), then a re-ranking pass. Unsupervised (random rotation + sign), numpy-only, one knob
(``beam``). It stores compact codes rather than full vectors, so it sits on the organisation axis
at a fraction of a graph's footprint. Research entry to the BitBudget index benchmark; see the
projection-quantisation-organisation survey for the motivation.

This reference implementation packs into a single uint64, so ``n_bits <= 64``.
"""
import numpy as np


class BitTrieIndex:
    def __init__(self, n_bits=64, seed=0, rotation="random", code_bits=None):
        assert n_bits <= 64, "this reference packs into one uint64; use a word-array for >64 bits"
        self.b = n_bits
        self.seed = seed
        assert rotation in ("random", "pca"), rotation
        # code_bits (multiple of 64, > n_bits): also store a longer code per doc and use it as a Hamming
        # prefilter before the float re-rank. Routing still uses only the top n_bits.
        assert code_bits is None or (code_bits % 64 == 0 and code_bits > n_bits), code_bits
        self.W = code_bits or n_bits
        self.rotation = rotation                                  # "pca": centre + top-b principal directions

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        rng = np.random.RandomState(self.seed)
        if self.rotation == "pca":
            assert self.W <= d, "pca rotation needs code_bits <= dim"
            self.mu = X.mean(axis=0)
            samp = X[rng.choice(n, min(n, 20000), replace=False)] - self.mu
            R = np.linalg.svd(samp, full_matrices=False)[2][: self.W].T.astype(np.float32)
        else:
            self.mu = np.zeros(d, np.float32)
            R = rng.randn(d, self.W).astype(np.float32)           # unsupervised random rotation (RaBitQ)
            R, _ = np.linalg.qr(R) if self.W <= d else (R, None)
        P = (X - self.mu) @ R
        order = np.argsort(-P.var(axis=0))                        # coarse-to-fine: MSB = highest variance
        self.Rfull = R[:, order]
        self.R = self.Rfull[:, : self.b]                          # routing projection (top-variance bits)
        P = P[:, order]
        bits = (P > 0).astype(np.uint64)
        w = (np.uint64(1) << np.arange(self.b - 1, -1, -1, dtype=np.uint64))
        codes = bits[:, : self.b] @ w
        order_idx = np.argsort(codes, kind="stable")
        self.ext = None
        if self.W > self.b:                                       # packed W-bit codes, in trie-sorted order
            self.ext = np.ascontiguousarray(self._pack(bits)[order_idx])
        self.codes = codes[order_idx]                             # sorted: the implicit trie
        self.docids = order_idx.astype(np.int64)
        self.X = X                                                # kept only for optional float re-rank (cold)
        self.n, self.d = n, d
        return self

    @staticmethod
    def _pack(bits):
        """(m, W) 0/1 -> (m, W/64) uint64, MSB first within each word."""
        m, W = bits.shape
        wt = (np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64))
        return np.stack([bits[:, i:i + 64].astype(np.uint64) @ wt for i in range(0, W, 64)], axis=1)

    def _descend(self, qbits, qconf, beam, depth):
        """Confidence-ordered beam over the sorted codes. Stop at ``depth`` bits so each surviving
        prefix is still a fat bucket (the coarse-to-fine / anytime property)."""
        b = self.b
        state = [(0.0, 0, self.n, 0, 0)]                          # (cost, lo, hi, prefix_int, depth)
        for t in range(depth):
            shift = b - t - 1
            nxt = []
            for cost, lo, hi, pre, _ in state:
                thresh = (pre << (b - t)) | (1 << shift)
                m = lo + int(np.searchsorted(self.codes[lo:hi], np.uint64(thresh), "left"))
                qb = int(qbits[t]); c = float(qconf[t])
                left = (lo, m, (pre << 1))
                right = (m, hi, (pre << 1) | 1)
                for child_bit, (clo, chi, cpre) in ((0, left), (1, right)):
                    if chi <= clo:
                        continue
                    add = 0.0 if child_bit == qb else c           # sibling costs its margin
                    nxt.append((cost + add, clo, chi, cpre, t + 1))
            nxt.sort(key=lambda e: e[0])
            state = nxt[:beam]
        return state

    def query(self, q, topk=10, beam=64, depth=28, rerank="float"):
        q = np.asarray(q, dtype=np.float32)
        p = (q - self.mu) @ self.R
        qbits = (p > 0).astype(int)
        qconf = np.abs(p)
        ranges = self._descend(qbits, qconf, beam, depth)
        cand = np.concatenate([self.docids[lo:hi] for _, lo, hi, _, _ in ranges]) if ranges else np.empty(0, np.int64)
        if cand.size == 0:
            return cand
        if rerank == "float":                                     # DiskANN-style: cold float for re-rank only
            s = self.X[cand] @ q
        else:                                                     # asymmetric: float query vs sign code, no float
            shifts = np.arange(self.b - 1, -1, -1, dtype=np.uint64)
            sign = np.where(((self.codes_full(cand)[:, None] >> shifts) & np.uint64(1)) > 0, 1.0, -1.0)
            s = sign @ ((q - self.mu) @ self.R)
        return cand[np.argsort(-s)[:topk]]

    def query_batch(self, Q, topk=10, beam=64, depth=28, rerank=None):
        """Top-k for every query in Q. Uses the compiled C kernel when available (one to two
        orders of magnitude faster), else falls back to the pure-Python per-query path."""
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        depth = min(depth, self.b)
        from ._bittrie_build import get_lib
        lib = get_lib()
        if lib is None:
            return np.array([self.query(q, topk, beam, depth, rerank="float") for q in Q])
        import ctypes
        qproj = np.ascontiguousarray((Q - self.mu) @ self.R, dtype=np.float32)
        Xf = np.ascontiguousarray(self.X, dtype=np.float32)
        out = np.full((len(Q), topk), -1, dtype=np.int64)
        cast = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))
        if self.ext is not None:
            qext = np.ascontiguousarray(self._pack((((Q - self.mu) @ self.Rfull) > 0).astype(np.uint64)))
            lib.bt_query_batch_ext(
                cast(self.codes, ctypes.c_uint64), cast(self.docids, ctypes.c_int64),
                cast(Xf, ctypes.c_float), cast(qproj, ctypes.c_float), cast(Q, ctypes.c_float),
                cast(self.ext, ctypes.c_uint64), cast(qext, ctypes.c_uint64), self.W // 64, int(rerank or 0),
                self.n, self.d, self.b, len(Q), int(depth), int(beam), int(topk), cast(out, ctypes.c_int64))
            return out
        lib.bt_query_batch(
            cast(self.codes, ctypes.c_uint64), cast(self.docids, ctypes.c_int64),
            cast(Xf, ctypes.c_float), cast(qproj, ctypes.c_float), cast(Q, ctypes.c_float),
            self.n, self.d, self.b, len(Q), int(depth), int(beam), int(topk),
            cast(out, ctypes.c_int64))
        return out

    def codes_full(self, cand):
        inv = np.empty(self.n, np.int64); inv[self.docids] = np.arange(self.n)
        return self.codes[inv[cand]]

    def index_bytes(self):
        """Bytes that must be in RAM to route (codes + leaf ids); the trie is implicit in the sort."""
        return self.codes.nbytes + self.docids.nbytes + (self.ext.nbytes if self.ext is not None else 0)
