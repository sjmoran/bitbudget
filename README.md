# BitBudget

**How much retrieval quality do you keep per byte?**

BitBudget is a small, reproducible benchmark for **embedding compression**. Give it an
embedder and a corpus and it reports the retrieval quality (nDCG@10, recall@10) that each
compression method retains against the **bytes it stores per vector** — the recall‑per‑byte
frontier that every RAG and vector‑database deployment actually lives on.

It is the companion benchmark to the survey *“Projection and Quantisation: A Unifying View of
Learning to Hash, from Random Projections to the RAG Era”* and exists to answer one question
that today is mostly answered by vendor blog posts: **when you binarise / int8 / RaBitQ /
product‑quantise / Matryoshka‑truncate your embeddings, what do you actually lose?**

## The headline finding

> **Bits beat dimensions.** Spending a fixed byte budget on *more coarsely quantised*
> coordinates beats spending it on *fewer full‑precision* coordinates, at every budget and
> for every embedder we have tried. One‑bit codes with a cheap re‑ranking pass are **32×
> smaller than float at no measurable loss**.

```
mxbai‑embed‑large (1024‑d), mean over 4 BEIR corpora
  binary+rerank      128 B   nDCG 0.509   100% of float   ← 32× smaller, lossless
  pq                 128 B   nDCG 0.488    96%
  rabitq             128 B   nDCG 0.487    96%
  matryoshka        1024 B   nDCG 0.439    86%             ← 4× smaller, projection axis
  float32           4096 B   nDCG 0.508   100%
```

See **[LEADERBOARD.md](LEADERBOARD.md)** for the full table.

## Install

```bash
pip install bitbudget            # evaluation only (numpy)
pip install "bitbudget[all]"     # + sentence-transformers (embedding) + faiss
```

## Quickstart

```bash
bitbudget methods                                   # list compression methods
bitbudget run --embedder mxbai --corpus scifact     # embed + evaluate, print a results card
bitbudget leaderboard results/card_*.json           # render a markdown leaderboard
```

`run` embeds (torch) and evaluates (numpy) in one process. The corpora auto‑download.

> **macOS note.** torch and faiss each bundle their own OpenMP runtime and crash if imported
> in the same process. The core methods are numpy‑only, so `run` is safe; if you add a
> faiss‑backed method, run `bitbudget embed` (torch) and `bitbudget eval` (numpy/faiss)
> as separate processes.

## The protocol (frozen, so results are comparable)

- **Corpora:** the BEIR subsets `scifact`, `nfcorpus`, `arguana`, `fiqa` (small enough to run
  on a laptop, diverse enough to be honest). Numbers are the mean over corpora; `±` is the
  standard deviation across them.
- **Metrics:** `nDCG@10` against the graded BEIR judgements, and `recall@10` against the exact
  floating‑point neighbours. `% of float` is nDCG relative to the uncompressed embedding.
- **Memory:** bytes stored per document vector (`4D` float, `D` int8, `D/8` binary, `M` for an
  `M`‑byte product code, `4·dim` for a truncated/PCA‑reduced vector).
- **Embedders:** `minilm` (384‑d) and `mxbai` (1024‑d, Matryoshka) ship built in.

## Add your method in five lines

This is the point of the benchmark: drop in your compressor and it is scored against every
built‑in on the same protocol.

```python
from bitbudget import method
import numpy as np

@method("my-2bit", bits=2)
def my_2bit(demb, qemb):
    codes = my_quantise(demb)                       # your compression
    scores = qemb @ my_reconstruct(codes).T         # (queries x docs) similarity
    return scores, demb.shape[1] * 2 / 8            # scores, bytes per stored vector
```

```bash
bitbudget run --embedder mxbai --corpus scifact --methods my-2bit binary+rerank float32
```

Then open a pull request adding your row to [LEADERBOARD.md](LEADERBOARD.md). See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Cite

If BitBudget helps your work, please cite the survey:

```bibtex
@article{moran2025projection,
  title   = {Projection and Quantisation: A Unifying View of Learning to Hash,
             from Random Projections to the RAG Era},
  author  = {Moran, Sean},
  journal = {arXiv preprint arXiv:2510.04127},
  year    = {2025}
}
```

MIT licensed.
