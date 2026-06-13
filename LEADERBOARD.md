# BitBudget leaderboard

<!-- Generated from site/data.json by make_leaderboard.py — do not edit by hand. -->

**Live leaderboard:** https://sjmoran.github.io/bitbudget/ (regenerated from `site/data.json` on merge)
**Paper:** https://arxiv.org/abs/2510.04127

nDCG@10 retained per byte, mean over three BEIR corpora (scifact, nfcorpus, arguana). '% of float' is nDCG@10 relative to the uncompressed embedding; '±' is the standard deviation across the corpora. Every row is reproducible with `bitbudget run --embedder <name> --corpus scifact nfcorpus arguana`.

**To submit:** register your method (`@method(...)`), run
`bitbudget run --embedder <name> --corpus scifact nfcorpus arguana`, and open a PR adding
your row with the generated results card attached. See [CONTRIBUTING.md](CONTRIBUTING.md).

## text-embedding-3-small (1536-d · hosted · Matryoshka)

| Method | Axis | Bytes/vec | nDCG@10 | % of float |
|---|---|---|---|---|
| binary | quantisation | 192 | 0.488 | 95.8 ± 0.9 |
| binary+rerank | quantisation | 192 | 0.509 | **100.0 ± 0.0** |
| binary-mean | quantisation | 192 | 0.490 | 95.7 ± 3.3 |
| pq | quantisation | 192 | 0.496 | 97.6 ± 1.2 |
| rabitq | quantisation | 192 | 0.496 | 97.8 ± 0.7 |
| int4 | quantisation | 768 | 0.509 | **100.1 ± 0.6** |
| int8 | quantisation | 1536 | 0.508 | **100.0 ± 0.1** |
| matryoshka | projection | 1536 | 0.491 | 96.1 ± 1.6 |
| pca | projection | 1536 | 0.483 | 94.7 ± 3.7 |
| float16 | quantisation | 3072 | 0.508 | **100.0 ± 0.0** |
| float32 | none | 6144 | 0.508 | 100 |

## text-embedding-3-large (3072-d · hosted · Matryoshka)

| Method | Axis | Bytes/vec | nDCG@10 | % of float |
|---|---|---|---|---|
| binary | quantisation | 384 | 0.536 | 98.8 ± 0.5 |
| binary+rerank | quantisation | 384 | 0.543 | **100.0 ± 0.0** |
| binary-mean | quantisation | 384 | 0.527 | 96.9 ± 2.6 |
| pq | quantisation | 384 | 0.535 | 98.6 ± 0.6 |
| rabitq | quantisation | 384 | 0.535 | 98.6 ± 0.5 |
| int4 | quantisation | 1536 | 0.541 | **99.6 ± 0.2** |
| int8 | quantisation | 3072 | 0.543 | **100.0 ± 0.0** |
| matryoshka | projection | 3072 | 0.527 | 96.9 ± 1.3 |
| pca | projection | 3072 | 0.523 | 96.2 ± 3.2 |
| float16 | quantisation | 6144 | 0.543 | **100.0 ± 0.0** |
| float32 | none | 12288 | 0.543 | 100 |

## mxbai-embed-large (1024-d · Matryoshka)

| Method | Axis | Bytes/vec | nDCG@10 | % of float |
|---|---|---|---|---|
| binary | quantisation | 128 | 0.491 | 92.6 ± 1.9 |
| binary+rerank | quantisation | 128 | 0.530 | **100.2 ± 0.2** |
| binary-mean | quantisation | 128 | 0.500 | 94.5 ± 0.8 |
| pq | quantisation | 128 | 0.514 | 97.1 ± 0.9 |
| rabitq | quantisation | 128 | 0.511 | 96.5 ± 1.1 |
| int4 | quantisation | 512 | 0.527 | **99.8 ± 0.4** |
| int8 | quantisation | 1024 | 0.530 | **100.2 ± 0.1** |
| matryoshka | projection | 1024 | 0.466 | 88.4 ± 1.4 |
| pca | projection | 1024 | 0.480 | 89.7 ± 4.8 |
| float16 | quantisation | 2048 | 0.528 | **100.0 ± 0.0** |
| float32 | none | 4096 | 0.528 | 100 |

## bge-base-en-v1.5 (768-d)

| Method | Axis | Bytes/vec | nDCG@10 | % of float |
|---|---|---|---|---|
| binary | quantisation | 96 | 0.466 | 89.2 ± 0.3 |
| binary+rerank | quantisation | 96 | 0.522 | **99.8 ± 0.2** |
| binary-mean | quantisation | 96 | 0.473 | 90.4 ± 1.4 |
| pq | quantisation | 96 | 0.506 | 96.6 ± 1.1 |
| rabitq | quantisation | 96 | 0.492 | 93.7 ± 1.5 |
| int4 | quantisation | 384 | 0.520 | 99.3 ± 0.3 |
| int8 | quantisation | 768 | 0.522 | **99.9 ± 0.1** |
| matryoshka | projection | 768 | 0.430 | 81.4 ± 3.0 |
| pca | projection | 768 | 0.462 | 87.2 ± 4.8 |
| float16 | quantisation | 1536 | 0.523 | **100.0 ± 0.0** |
| float32 | none | 3072 | 0.523 | 100 |

## gte-base (768-d)

| Method | Axis | Bytes/vec | nDCG@10 | % of float |
|---|---|---|---|---|
| binary | quantisation | 96 | 0.453 | 87.0 ± 2.7 |
| binary+rerank | quantisation | 96 | 0.515 | **99.7 ± 0.6** |
| binary-mean | quantisation | 96 | 0.483 | 93.0 ± 2.8 |
| pq | quantisation | 96 | 0.481 | 93.1 ± 1.2 |
| rabitq | quantisation | 96 | 0.484 | 93.1 ± 1.8 |
| int4 | quantisation | 384 | 0.511 | 98.6 ± 1.2 |
| int8 | quantisation | 768 | 0.515 | **99.8 ± 0.1** |
| matryoshka | projection | 768 | 0.363 | 69.0 ± 5.2 |
| pca | projection | 768 | 0.471 | 90.4 ± 3.9 |
| float16 | quantisation | 1536 | 0.517 | **100.0 ± 0.0** |
| float32 | none | 3072 | 0.517 | 100 |

## e5-base-v2 (768-d)

| Method | Axis | Bytes/vec | nDCG@10 | % of float |
|---|---|---|---|---|
| binary | quantisation | 96 | 0.253 | 52.7 ± 7.5 |
| binary+rerank | quantisation | 96 | 0.419 | 90.4 ± 2.2 |
| binary-mean | quantisation | 96 | 0.401 | 85.7 ± 3.8 |
| pq | quantisation | 96 | 0.419 | 90.5 ± 2.5 |
| rabitq | quantisation | 96 | 0.377 | 81.1 ± 3.1 |
| int4 | quantisation | 384 | 0.453 | 98.2 ± 1.5 |
| int8 | quantisation | 768 | 0.460 | **100.0 ± 0.2** |
| matryoshka | projection | 768 | 0.234 | 51.5 ± 1.9 |
| pca | projection | 768 | 0.391 | 82.8 ± 5.4 |
| float16 | quantisation | 1536 | 0.460 | **100.0 ± 0.0** |
| float32 | none | 3072 | 0.461 | 100 |

## all-MiniLM-L6-v2 (384-d)

| Method | Axis | Bytes/vec | nDCG@10 | % of float |
|---|---|---|---|---|
| binary | quantisation | 48 | 0.398 | 89.1 ± 1.3 |
| binary+rerank | quantisation | 48 | 0.443 | **99.8 ± 0.2** |
| binary-mean | quantisation | 48 | 0.388 | 86.8 ± 4.5 |
| pq | quantisation | 48 | 0.426 | 95.3 ± 1.5 |
| rabitq | quantisation | 48 | 0.416 | 93.6 ± 1.7 |
| int4 | quantisation | 192 | 0.446 | **100.1 ± 1.0** |
| int8 | quantisation | 384 | 0.445 | **100.1 ± 0.1** |
| matryoshka | projection | 384 | 0.349 | 78.0 ± 1.9 |
| pca | projection | 384 | 0.379 | 84.4 ± 6.8 |
| float16 | quantisation | 768 | 0.444 | **100.0 ± 0.0** |
| float32 | none | 1536 | 0.444 | 100 |

**Reading it.** At a matched footprint the quantisation-axis methods (binary, PQ, RaBitQ)
dominate the projection-axis ones (Matryoshka, PCA): bits beat dimensions. A one-bit code with a
re-ranking pass (`binary+rerank`) is essentially lossless at 32x compression on every embedder
except the strongly off-centre e5-base-v2, where a zero threshold misallocates its bits;
re-centring the threshold at each coordinate's mean (`binary-mean`) recovers it. `% of float` is
nDCG@10 relative to the uncompressed embedding; `±` is the standard deviation across the corpora.
