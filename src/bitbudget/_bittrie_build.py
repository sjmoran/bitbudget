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
import platform
import subprocess
import sys

_LIB = None
_TRIED = False


def _cache_dir():
    d = os.environ.get("BITBUDGET_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "bitbudget")
    os.makedirs(d, exist_ok=True)
    return d


def _brew_libomp():
    """Return the Homebrew libomp prefix on macOS, or None."""
    home = "/opt/homebrew" if platform.machine() == "arm64" else "/usr/local"   # Homebrew prefix matching our arch
    if os.path.isdir(f"{home}/opt/libomp"):
        return f"{home}/opt/libomp"
    try:
        p = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True)
        prefix = p.stdout.strip()
        if p.returncode == 0 and prefix and os.path.isdir(prefix):
            return prefix
    except Exception:
        pass
    return None


def _variants():
    """(cflags, lflags) compile attempts, OpenMP first, single-thread C last."""
    v = []
    omp = _brew_libomp()
    if omp:                                                    # macOS clang + Homebrew libomp (try first)
        v.append((["-Xpreprocessor", "-fopenmp", f"-I{omp}/include"],
                  [f"-L{omp}/lib", "-lomp", f"-Wl,-rpath,{omp}/lib"]))
    v.append((["-fopenmp"], ["-fopenmp"]))                     # gcc / OpenMP-capable clang (Linux)
    v.append((["-fopenmp-simd"], []))                          # no OpenMP runtime: single-threaded C, simd pragma only
    v.append(([], []))                                         # last resort: plain C
    return v


def _arch_flags():
    """macOS: build for the running interpreter's arch (an x86_64 Python under Rosetta on an arm64
    host cannot dlopen the compiler's default arm64 output)."""
    if sys.platform == "darwin" and platform.machine() in ("x86_64", "arm64"):
        return ["-arch", platform.machine()]
    return []


def _set_sig(lib):
    f = lib.bt_query_batch
    f.restype = None
    P = ctypes.POINTER
    f.argtypes = [P(ctypes.c_uint64), P(ctypes.c_int64), P(ctypes.c_float),
                  P(ctypes.c_float), P(ctypes.c_float),
                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int, ctypes.c_int, P(ctypes.c_int64)]
    g = lib.bt_query_batch_ext
    g.restype = None
    g.argtypes = [P(ctypes.c_uint64), P(ctypes.c_int64), P(ctypes.c_float),
                  P(ctypes.c_float), P(ctypes.c_float),
                  P(ctypes.c_uint64), P(ctypes.c_uint64), ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int, ctypes.c_int, P(ctypes.c_int64)]
    return lib


def get_lib():
    """Return the loaded ctypes library exposing bt_query_batch, or None if unavailable.

    Tries to build a multithreaded (OpenMP) kernel first, falling back to single-threaded C,
    then to None (the caller uses the numpy/Python path). The compiled library is cached by
    source hash; set BITBUDGET_NO_OMP=1 to force the single-threaded build.
    """
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    src = os.path.join(os.path.dirname(__file__), "_bittrie.c")
    if not os.path.exists(src):
        return None
    h = hashlib.sha1(open(src, "rb").read()).hexdigest()[:12]
    so = os.path.join(_cache_dir(), f"_bittrie_{h}.so")
    if os.path.exists(so):                                     # cached: load directly
        try:
            _LIB = _set_sig(ctypes.CDLL(so)); return _LIB
        except Exception:
            try: os.remove(so)
            except Exception: pass
    cc = os.environ.get("CC") or "cc"
    variants = [(["-fopenmp-simd"], [])] if os.environ.get("BITBUDGET_NO_OMP") else _variants()
    for cflags, lflags in variants:
        cmd = [cc, "-O3", "-shared", "-fPIC", *_arch_flags(), *cflags, "-o", so, src, *lflags, "-lm"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception:
            continue
        try:                                                  # must actually load (libomp present at runtime)
            _LIB = _set_sig(ctypes.CDLL(so)); return _LIB
        except Exception:
            try: os.remove(so)
            except Exception: pass
    return None
