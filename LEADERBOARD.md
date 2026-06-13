# BitBudget leaderboard

**Live leaderboard:** https://sjmoran.github.io/bitbudget/ (regenerated from `site/data.json` on merge)
**Paper:** https://arxiv.org/abs/2510.04127

Retrieval quality retained per byte, mean over the BEIR corpora `scifact`, `nfcorpus`,
`arguana`, `fiqa`. `% of float` is nDCG@10 relative to the uncompressed embedding; `±` is the
standard deviation across the four corpora. Sorted by footprint.

**To submit:** register your method (`@method(...)`), run
`bitbudget run --embedder mxbai --corpus scifact nfcorpus arguana fiqa`, and open a PR adding
your row with the generated results card attached. See [CONTRIBUTING.md](CONTRIBUTING.md).

## mxbai-embed-large (1024-d, Matryoshka)

| Method | Axis | Bytes/vec | nDCG@10 | % of float |
|---|---|---|---|---|
| binary | quantisation | 128 | 0.469 | 92 ± 2 |
| binary+rerank | quantisation | 128 | 0.509 | **100 ± 0** |
| pq | quantisation | 128 | 0.488 | 96 ± 2 |
| rabitq | quantisation | 128 | 0.487 | 96 ± 1 |
| int8 | quantisation | 1024 | 0.509 | 100 ± 0 |
| matryoshka | projection | 1024 | 0.439 | 86 ± 4 |
| pca | projection | 1024 | 0.450 | 88 ± 6 |
| float32 | none | 4096 | 0.508 | 100 |

## all-MiniLM-L6-v2 (384-d)

| Method | Axis | Bytes/vec | nDCG@10 | % of float |
|---|---|---|---|---|
| binary | quantisation | 48 | 0.375 | 88 ± 3 |
| binary+rerank | quantisation | 48 | 0.424 | **100 ± 0** |
| pq | quantisation | 48 | 0.401 | 94 ± 3 |
| rabitq | quantisation | 48 | 0.400 | 94 ± 2 |
| int8 | quantisation | 384 | 0.426 | 100 ± 0 |
| matryoshka | projection | 384 | 0.324 | 75 ± 5 |
| pca | projection | 384 | 0.358 | 83 ± 6 |
| float32 | none | 1536 | 0.425 | 100 |

**Reading it.** At a matched footprint the quantisation‑axis methods (binary, PQ, RaBitQ)
dominate the projection‑axis ones (Matryoshka, PCA): bits beat dimensions. A one‑bit code with
a re‑ranking pass is lossless at 32× compression on both embedders, and the larger embedding
binarises *better* (92% vs 88%) because its information is spread across more coordinates.
