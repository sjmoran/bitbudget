"""BitBudget: how much retrieval quality do you keep per byte?

A reproducible benchmark for embedding compression. Given an embedder and a corpus, it
measures the retrieval quality (nDCG@10, recall@10) retained by each compression method
against the bytes it stores per vector -- the recall-per-byte frontier.

Add your own method in five lines::

    from bitbudget import method
    import numpy as np

    @method("my-2bit", bits=2)
    def my_2bit(demb, qemb):
        # return (query x doc similarity scores, bytes per stored vector)
        codes = my_quantise(demb)
        return qemb @ my_dequantise(codes).T, demb.shape[1] * 2 / 8

Then ``bitbudget run --embedder mxbai --corpus scifact`` scores it alongside the built-ins.
"""
from .methods import method, METHODS, list_methods
from .embedders import embedder, EMBEDDERS, list_embedders
from .eval import evaluate
from .indexes import index, INDEXES, list_indexes, bench_indexes

__version__ = "0.3.0"
__all__ = ["method", "METHODS", "list_methods", "embedder", "EMBEDDERS",
           "list_embedders", "evaluate", "index", "INDEXES", "list_indexes",
           "bench_indexes", "__version__"]
