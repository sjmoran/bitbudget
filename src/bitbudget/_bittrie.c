/* Fast batched bit-trie query: coarse-to-fine beam descent over sorted codes, then a
 * full-precision re-ranking pass, looped over all queries in C. Plain C (no Python.h);
 * loaded via ctypes from _bittrie_build.py and compiled on demand. Falls back to the
 * numpy/Python path in bittrie.py if no compiler is available. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _MSC_VER
#define POPCNT64(x) ((int)__popcnt64(x))
#else
#define POPCNT64(x) __builtin_popcountll(x)
#endif

typedef struct { double cost; long lo, hi; unsigned long long pre; } Node;

static int cmp_node(const void *a, const void *b) {
    double d = ((const Node *)a)->cost - ((const Node *)b)->cost;
    return (d > 0) - (d < 0);
}

static long lower_bound_u64(const uint64_t *a, long lo, long hi, uint64_t key) {
    while (lo < hi) { long mid = lo + (hi - lo) / 2; if (a[mid] < key) lo = mid + 1; else hi = mid; }
    return lo;
}

/* out: nq*k int64 of original doc ids (top-k by inner product), -1 padding if fewer found. */
/* Each query is independent and writes a disjoint slice of `out`, so the loop parallelises
 * cleanly. Scratch buffers are allocated per thread inside the parallel region; when compiled
 * without OpenMP the `omp` pragmas are ignored and this runs as one block, single-threaded. */
void bt_query_batch(const uint64_t *codes, const int64_t *docids,
                    const float *Xf, const float *qproj, const float *qf,
                    int n, int d, int b, int nq, int depth, int beam, int k,
                    int64_t *out) {
    if (depth > b) depth = b;
    if (beam < 1) beam = 1;
    int cap = 2 * beam + 4;

    #pragma omp parallel
    {
        Node *cur = (Node *)malloc(sizeof(Node) * cap);
        Node *nxt = (Node *)malloc(sizeof(Node) * 2 * cap);
        double *bscore = (double *)malloc(sizeof(double) * k);
        long *bid = (long *)malloc(sizeof(long) * k);

        #pragma omp for schedule(static)
        for (int qi = 0; qi < nq; qi++) {
            const float *qp = qproj + (size_t)qi * b;
            const float *qv = qf + (size_t)qi * d;
            int ncur = 1;
            cur[0].cost = 0.0; cur[0].lo = 0; cur[0].hi = n; cur[0].pre = 0ULL;

            for (int t = 0; t < depth; t++) {
                int shift = b - t - 1;
                int qb = qp[t] > 0.0f ? 1 : 0;
                double c = fabs((double)qp[t]);
                int nn = 0;
                for (int i = 0; i < ncur; i++) {
                    long lo = cur[i].lo, hi = cur[i].hi;
                    unsigned long long pre = cur[i].pre;
                    unsigned long long hi_part = (b - t) >= 64 ? 0ULL : (pre << (b - t));
                    unsigned long long thresh = hi_part | (1ULL << shift);
                    long m = lower_bound_u64(codes, lo, hi, (uint64_t)thresh);
                    if (m > lo) { nxt[nn].cost = cur[i].cost + (qb == 0 ? 0.0 : c);
                                  nxt[nn].lo = lo; nxt[nn].hi = m; nxt[nn].pre = (pre << 1); nn++; }
                    if (hi > m) { nxt[nn].cost = cur[i].cost + (qb == 1 ? 0.0 : c);
                                  nxt[nn].lo = m; nxt[nn].hi = hi; nxt[nn].pre = (pre << 1) | 1ULL; nn++; }
                }
                qsort(nxt, nn, sizeof(Node), cmp_node);
                int keep = nn < beam ? nn : beam;
                memcpy(cur, nxt, sizeof(Node) * keep);
                ncur = keep;
            }

            for (int j = 0; j < k; j++) { bscore[j] = -1e300; bid[j] = -1; }
            for (int i = 0; i < ncur; i++) {
                for (long j = cur[i].lo; j < cur[i].hi; j++) {
                    long id = (long)docids[j];
                    const float *xr = Xf + (size_t)id * d;
                    float sf = 0.0f;                  /* float accumulate + simd: ~4-8x faster than the scalar double loop */
                    #pragma omp simd reduction(+:sf)
                    for (int dd = 0; dd < d; dd++) sf += qv[dd] * xr[dd];
                    double s = (double)sf;
                    int mn = 0;
                    for (int t2 = 1; t2 < k; t2++) if (bscore[t2] < bscore[mn]) mn = t2;
                    if (s > bscore[mn]) { bscore[mn] = s; bid[mn] = id; }
                }
            }
            for (int a = 0; a < k; a++) {        /* selection-sort the k winners, score descending */
                int mx = a;
                for (int b2 = a + 1; b2 < k; b2++) if (bscore[b2] > bscore[mx]) mx = b2;
                double ts = bscore[a]; bscore[a] = bscore[mx]; bscore[mx] = ts;
                long ti = bid[a]; bid[a] = bid[mx]; bid[mx] = ti;
            }
            for (int j = 0; j < k; j++) out[(size_t)qi * k + j] = bid[j];
        }
        free(cur); free(nxt); free(bscore); free(bid);
    }
}


/* Same descent as bt_query_batch, but candidates are first ranked by Hamming distance between a
 * W-bit query code and a W-bit doc code (nw uint64 words each, `ext` in trie-sorted order), and only
 * the `rerank` closest go to the float dot product. rerank <= 0 reranks every candidate (== old path).
 * Ties at the cut-off are all kept, so the output is deterministic. */
void bt_query_batch_ext(const uint64_t *codes, const int64_t *docids,
                        const float *Xf, const float *qproj, const float *qf,
                        const uint64_t *ext, const uint64_t *qext, int nw, int rerank,
                        int n, int d, int b, int nq, int depth, int beam, int k,
                        int64_t *out) {
    if (depth > b) depth = b;
    if (beam < 1) beam = 1;
    int cap = 2 * beam + 4;
    int maxh = 64 * nw;

    #pragma omp parallel
    {
        Node *cur = (Node *)malloc(sizeof(Node) * cap);
        Node *nxt = (Node *)malloc(sizeof(Node) * 2 * cap);
        double *bscore = (double *)malloc(sizeof(double) * k);
        long *bid = (long *)malloc(sizeof(long) * k);
        long *hist = (long *)malloc(sizeof(long) * (maxh + 1));
        long hcap = 1024;
        uint16_t *ham = (uint16_t *)malloc(sizeof(uint16_t) * hcap);

        #pragma omp for schedule(static)
        for (int qi = 0; qi < nq; qi++) {
            const float *qp = qproj + (size_t)qi * b;
            const float *qv = qf + (size_t)qi * d;
            const uint64_t *qe = qext + (size_t)qi * nw;
            int ncur = 1;
            cur[0].cost = 0.0; cur[0].lo = 0; cur[0].hi = n; cur[0].pre = 0ULL;

            for (int t = 0; t < depth; t++) {
                int shift = b - t - 1;
                int qb = qp[t] > 0.0f ? 1 : 0;
                double c = fabs((double)qp[t]);
                int nn = 0;
                for (int i = 0; i < ncur; i++) {
                    long lo = cur[i].lo, hi = cur[i].hi;
                    unsigned long long pre = cur[i].pre;
                    unsigned long long hi_part = (b - t) >= 64 ? 0ULL : (pre << (b - t));
                    unsigned long long thresh = hi_part | (1ULL << shift);
                    long m = lower_bound_u64(codes, lo, hi, (uint64_t)thresh);
                    if (m > lo) { nxt[nn].cost = cur[i].cost + (qb == 0 ? 0.0 : c);
                                  nxt[nn].lo = lo; nxt[nn].hi = m; nxt[nn].pre = (pre << 1); nn++; }
                    if (hi > m) { nxt[nn].cost = cur[i].cost + (qb == 1 ? 0.0 : c);
                                  nxt[nn].lo = m; nxt[nn].hi = hi; nxt[nn].pre = (pre << 1) | 1ULL; nn++; }
                }
                qsort(nxt, nn, sizeof(Node), cmp_node);
                int keep = nn < beam ? nn : beam;
                memcpy(cur, nxt, sizeof(Node) * keep);
                ncur = keep;
            }

            /* pass 1: Hamming distance for every candidate, histogrammed */
            long ncand = 0;
            memset(hist, 0, sizeof(long) * (maxh + 1));
            for (int i = 0; i < ncur; i++) ncand += cur[i].hi - cur[i].lo;
            if (ncand > hcap) { hcap = ncand; ham = (uint16_t *)realloc(ham, sizeof(uint16_t) * hcap); }
            long p = 0;
            for (int i = 0; i < ncur; i++)
                for (long j = cur[i].lo; j < cur[i].hi; j++, p++) {
                    const uint64_t *de = ext + (size_t)j * nw;
                    int h = 0;
                    for (int w = 0; w < nw; w++) h += POPCNT64(de[w] ^ qe[w]);
                    ham[p] = (uint16_t)h; hist[h]++;
                }
            int tmax = maxh;
            if (rerank > 0 && rerank < ncand) {
                long cum = 0;
                for (tmax = 0; tmax <= maxh; tmax++) { cum += hist[tmax]; if (cum >= rerank) break; }
            }

            /* pass 2: float rerank of the survivors */
            for (int j = 0; j < k; j++) { bscore[j] = -1e300; bid[j] = -1; }
            p = 0;
            for (int i = 0; i < ncur; i++) {
                for (long j = cur[i].lo; j < cur[i].hi; j++, p++) {
                    if (ham[p] > tmax) continue;
                    long id = (long)docids[j];
                    const float *xr = Xf + (size_t)id * d;
                    float sf = 0.0f;
                    #pragma omp simd reduction(+:sf)
                    for (int dd = 0; dd < d; dd++) sf += qv[dd] * xr[dd];
                    double s = (double)sf;
                    int mn = 0;
                    for (int t2 = 1; t2 < k; t2++) if (bscore[t2] < bscore[mn]) mn = t2;
                    if (s > bscore[mn]) { bscore[mn] = s; bid[mn] = id; }
                }
            }
            for (int a = 0; a < k; a++) {
                int mx = a;
                for (int b2 = a + 1; b2 < k; b2++) if (bscore[b2] > bscore[mx]) mx = b2;
                double ts = bscore[a]; bscore[a] = bscore[mx]; bscore[mx] = ts;
                long ti = bid[a]; bid[a] = bid[mx]; bid[mx] = ti;
            }
            for (int j = 0; j < k; j++) out[(size_t)qi * k + j] = bid[j];
        }
        free(cur); free(nxt); free(bscore); free(bid); free(hist); free(ham);
    }
}
