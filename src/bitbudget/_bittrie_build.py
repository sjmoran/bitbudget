"""Compile the bit-trie C kernel on demand and load it via ctypes.

The package ships pure-Python on PyPI (the .c is data, not a built extension), so install never
needs a compiler. The first time the fast path is requested we compile _bittrie.c into a cached
shared library with the system C compiler; if no compiler is available we return None and the
caller falls back to the numpy/Python path. The compiled library is cached by source hash, so the
build happens once per machine.
"""
import ctypes
import hashlib
import os
import subprocess

_LIB = None
_TRIED = False


def _cache_dir():
    d = os.environ.get("BITBUDGET_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "bitbudget")
    os.makedirs(d, exist_ok=True)
    return d


def get_lib():
    """Return the loaded ctypes library exposing bt_query_batch, or None if unavailable."""
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    src = os.path.join(os.path.dirname(__file__), "_bittrie.c")
    if not os.path.exists(src):
        return None
    h = hashlib.sha1(open(src, "rb").read()).hexdigest()[:12]
    so = os.path.join(_cache_dir(), f"_bittrie_{h}.so")
    if not os.path.exists(so):
        cc = os.environ.get("CC") or "cc"
        cmd = [cc, "-O3", "-shared", "-fPIC", "-o", so, src, "-lm"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception:
            return None
    try:
        lib = ctypes.CDLL(so)
        f = lib.bt_query_batch
        f.restype = None
        P = ctypes.POINTER
        f.argtypes = [P(ctypes.c_uint64), P(ctypes.c_int64), P(ctypes.c_float),
                      P(ctypes.c_float), P(ctypes.c_float),
                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int, P(ctypes.c_int64)]
        _LIB = lib
    except Exception:
        _LIB = None
    return _LIB
