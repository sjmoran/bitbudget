# Contributing a method or embedder

BitBudget is designed to grow by contribution. The whole point is that you add *your*
compressor or embedder and it is scored against every built‑in on the same frozen protocol.

## Add a compression method

A method maps float document/query embeddings to a query‑by‑document similarity matrix under
your compression, and reports the bytes it stores per document vector.

```python
# bitbudget/contrib/my_method.py
from bitbudget import method
import numpy as np

@method("my-method", bits=2, axis="quantisation")   # axis: quantisation | projection | organisation
def my_method(demb, qemb):
    codes = my_quantise(demb)                        # compress the documents
    scores = qemb @ my_reconstruct(codes).T          # asymmetric: float query, decoded doc
    return scores, demb.shape[1] * 2 / 8             # (scores, bytes per stored vector)
```

Keep methods **numpy‑only** if you can, so they run in the same process as torch embedding.
If your method needs faiss, mark it and document that `embed` and `eval` must run separately.

## Add an embedder

```python
from bitbudget.embedders import _make_st_embedder
_make_st_embedder("my-embedder", "org/model-name", dim=768,
                  query_prompt="", matryoshka=False)
```

## Submit a leaderboard row

1. `bitbudget run --embedder mxbai --corpus scifact nfcorpus arguana fiqa --methods my-method binary+rerank float32`
2. Add your row to `LEADERBOARD.md` and attach the generated `results/card_*.json`.
3. Open a pull request. CI re‑runs the card to confirm it reproduces.

## Ground rules for comparability

- Report the **actual stored footprint** (the compact code), not the query‑time working set.
- A re‑ranking pass is allowed but must be declared; report the code size, not the re‑rank
  buffer.
- Do not tune on the test split. The four BEIR corpora are the evaluation set, full stop.
