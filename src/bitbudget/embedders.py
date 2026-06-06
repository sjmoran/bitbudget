"""Embedders. A named embedder turns a list of texts into a float32 matrix. These import
sentence-transformers (torch) and so must run in a process that does NOT also import faiss
(OpenMP clash on macOS); the CLI keeps embedding and faiss-backed evaluation in separate
processes. Register your own with ``@embedder``.
"""
EMBEDDERS = {}


def embedder(name, model=None, dim=None, query_prompt="", matryoshka=False):
    """Register an embedder. `query_prompt` is prepended to queries only (asymmetric models)."""
    def deco(fn):
        fn.bitbudget_name = name
        fn.model, fn.dim, fn.query_prompt, fn.matryoshka = model, dim, query_prompt, matryoshka
        EMBEDDERS[name] = fn
        return fn
    return deco


def list_embedders():
    return sorted(EMBEDDERS)


def _st(model_name, device, max_seq_length):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    if max_seq_length:
        m.max_seq_length = max_seq_length
    return m


def _make_st_embedder(name, model_name, dim, query_prompt, matryoshka, max_seq_length=256):
    @embedder(name, model=model_name, dim=dim, query_prompt=query_prompt, matryoshka=matryoshka)
    def fn(texts, is_query=False, device="cpu", batch_size=64):
        import numpy as np
        m = _st(model_name, device, max_seq_length)
        if is_query and query_prompt:
            texts = [query_prompt + t for t in texts]
        return m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)
    return fn


# built-in embedders used in the paper
_make_st_embedder("minilm", "sentence-transformers/all-MiniLM-L6-v2", 384, "", False)
_make_st_embedder("mxbai", "mixedbread-ai/mxbai-embed-large-v1", 1024,
                  "Represent this sentence for searching relevant passages: ", True)
