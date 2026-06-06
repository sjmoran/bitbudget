"""Embedders. A named embedder turns a list of texts into a float32 matrix. These import
sentence-transformers (torch) and so must run in a process that does NOT also import faiss
(OpenMP clash on macOS); the CLI keeps embedding and faiss-backed evaluation in separate
processes. Register your own with ``@embedder``.
"""
EMBEDDERS = {}


def embedder(name, model=None, dim=None, query_prompt="", matryoshka=False, doc_prompt=""):
    """Register an embedder. `query_prompt`/`doc_prompt` are prepended to queries/documents
    respectively (asymmetric models such as e5 and nomic want distinct prefixes for each)."""
    def deco(fn):
        fn.bitbudget_name = name
        fn.model, fn.dim, fn.query_prompt, fn.matryoshka = model, dim, query_prompt, matryoshka
        fn.doc_prompt = doc_prompt
        EMBEDDERS[name] = fn
        return fn
    return deco


def list_embedders():
    return sorted(EMBEDDERS)


_ST_CACHE = {}


def _st(model_name, device, max_seq_length):
    from sentence_transformers import SentenceTransformer
    key = (model_name, device)
    m = _ST_CACHE.get(key)                                # reuse across doc/query and corpora
    if m is None:
        m = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        _ST_CACHE[key] = m
    if max_seq_length:
        m.max_seq_length = max_seq_length
    return m


def _make_st_embedder(name, model_name, dim, query_prompt, matryoshka,
                      max_seq_length=256, doc_prompt=""):
    @embedder(name, model=model_name, dim=dim, query_prompt=query_prompt,
              matryoshka=matryoshka, doc_prompt=doc_prompt)
    def fn(texts, is_query=False, device="cpu", batch_size=64):
        import numpy as np
        m = _st(model_name, device, max_seq_length)
        prefix = query_prompt if is_query else doc_prompt
        if prefix:
            texts = [prefix + t for t in texts]
        return m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)
    return fn


# built-in embedders. The two from the paper, plus widely used open models so practitioners can
# see their own stack on the board. Prefixes follow each model's card: bge/mxbai use a query
# instruction only; e5 uses distinct query:/passage: prefixes; gte is symmetric (no prefix).
_make_st_embedder("minilm", "sentence-transformers/all-MiniLM-L6-v2", 384, "", False)
_make_st_embedder("mxbai", "mixedbread-ai/mxbai-embed-large-v1", 1024,
                  "Represent this sentence for searching relevant passages: ", True)
_make_st_embedder("bge-base", "BAAI/bge-base-en-v1.5", 768,
                  "Represent this sentence for searching relevant passages: ", False)
_make_st_embedder("e5-base", "intfloat/e5-base-v2", 768, "query: ", False,
                  doc_prompt="passage: ")
_make_st_embedder("gte-base", "thenlper/gte-base", 768, "", False)


def _make_openai_embedder(name, model_name, dim, matryoshka=True, batch=64):
    """A hosted OpenAI embedder, called over HTTPS so the package keeps no hard `openai`
    dependency. Needs OPENAI_API_KEY in the environment. text-embedding-3 is Matryoshka-trained,
    so prefix-truncation (the `matryoshka` method) is a fair projection baseline for it."""
    @embedder(name, model=model_name, dim=dim, query_prompt="", matryoshka=matryoshka)
    def fn(texts, is_query=False, device="cpu", batch_size=batch):
        import os, json, time, urllib.request, numpy as np
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit(f"OPENAI_API_KEY not set; required for embedder '{name}'.")
        vecs = []
        for i in range(0, len(texts), batch):
            # OpenAI rejects empty strings (400) and caps a request at ~300k tokens: replace
            # blanks with a placeholder and keep each input well under the per-input token limit.
            chunk = [(t.replace("\n", " ").strip()[:6000] or ".") for t in texts[i:i + batch]]
            body = json.dumps({"model": model_name, "input": chunk}).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/embeddings", data=body,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            for attempt in range(5):
                try:
                    r = json.load(urllib.request.urlopen(req, timeout=180)); break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))            # back off on rate limit / transient
            vecs.extend(d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"]))
        v = np.asarray(vecs, dtype=np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)   # cosine space
        return v
    return fn


_make_openai_embedder("openai-3-small", "text-embedding-3-small", 1536)
_make_openai_embedder("openai-3-large", "text-embedding-3-large", 3072)
