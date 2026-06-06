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
    def __init__(self, n_bits=64, seed=0):
        assert n_bits <= 64, "this reference packs into one uint64; use a word-array for >64 bits"
        self.b = n_bits
        self.seed = seed

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        rng = np.random.RandomState(self.seed)
        R = rng.randn(d, self.b).astype(np.float32)               # unsupervised random rotation (RaBitQ)
        R, _ = np.linalg.qr(R) if self.b <= d else (R, None)
        P = X @ R
        order = np.argsort(-P.var(axis=0))                        # coarse-to-fine: MSB = highest variance
        self.R = R[:, order]
        P = P[:, order]
        bits = (P > 0).astype(np.uint64)
        weights = (np.uint64(1) << np.arange(self.b - 1, -1, -1, dtype=np.uint64))
        codes = bits @ weights
        order_idx = np.argsort(codes, kind="stable")
        self.codes = codes[order_idx]                             # sorted: the implicit trie
        self.docids = order_idx.astype(np.int64)
        self.X = X                                                # kept only for optional float re-rank (cold)
        self.n, self.d = n, d
        return self

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
        p = q @ self.R
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
            s = sign @ (q @ self.R)
        return cand[np.argsort(-s)[:topk]]

    def codes_full(self, cand):
        inv = np.empty(self.n, np.int64); inv[self.docids] = np.arange(self.n)
        return self.codes[inv[cand]]

    def index_bytes(self):
        """Bytes that must be in RAM to route (codes + leaf ids); the trie is implicit in the sort."""
        return self.codes.nbytes + self.docids.nbytes
