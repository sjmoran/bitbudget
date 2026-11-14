"""BitBudget command line.

    bitbudget methods                              # list registered compression methods
    bitbudget embedders                            # list registered embedders
    bitbudget embed   --embedder mxbai --corpus scifact      # torch step -> cached embeddings
    bitbudget eval    --embedder mxbai --corpus scifact      # numpy step -> results card
    bitbudget run     --embedder mxbai --corpus scifact nfcorpus arguana fiqa   # embed + eval
    bitbudget leaderboard results/*.json           # aggregate cards into a markdown table
    bitbudget indexes                              # list registered indexes (organisation axis)
    bitbudget bench-index --synthetic 100000 128   # recall vs QPS vs bytes (flat/hnsw/ivfpq/bittrie)

`run` embeds (torch) and evaluates (numpy) in one process, which is safe because the core
methods use no faiss. The faiss-backed indexes in `bench-index` import faiss only when run, so
run it in its own process (no torch) to avoid the OpenMP clash.
"""
import argparse
import glob
import json
import os

import numpy as np

from . import datasets
from .eval import evaluate, aggregate
from .methods import list_methods, METHODS
from .indexes import bench_indexes, list_indexes, INDEXES


def _emb_path(cache, embedder, corpus):
    return os.path.join(os.path.expanduser(cache), f"emb_{embedder}_{corpus}.npz")


def cmd_embed(a):
    from .embedders import EMBEDDERS                      # lazy: imports torch
    emb = EMBEDDERS[a.embedder]
    for corpus in a.corpus:
        path = _emb_path(a.cache, a.embedder, corpus)
        if os.path.exists(path) and not getattr(a, "force", False):
            print(f"[embed] cached (skip) -> {path}")   # re-use; pass --force to recompute
            continue
        corp, queries, qrels = datasets.load(corpus, a.cache, a.split)
        cids, qids = list(corp), list(queries)
        print(f"[embed] {a.embedder} x {corpus}: {len(cids)} docs, {len(qids)} queries")
        demb = emb([corp[c] for c in cids], is_query=False, device=a.device, batch_size=a.batch_size)
        qemb = emb([queries[q] for q in qids], is_query=True, device=a.device, batch_size=a.batch_size)
        os.makedirs(os.path.expanduser(a.cache), exist_ok=True)
        cidset = set(cids)
        np.savez(_emb_path(a.cache, a.embedder, corpus), demb=demb, qemb=qemb,
                 cids=np.array(cids), qids=np.array(qids),
                 qrels=json.dumps({q: {d: s for d, s in qrels[q].items() if d in cidset} for q in qids}))
        print(f"[embed] cached -> {_emb_path(a.cache, a.embedder, corpus)}  {demb.shape}")


def _load_emb(a, corpus):
    z = np.load(_emb_path(a.cache, a.embedder, corpus), allow_pickle=True)
    return z["demb"], z["qemb"], z["cids"], z["qids"], json.loads(str(z["qrels"]))


def cmd_eval(a, _embed_first=False):
    methods = a.methods or list_methods()
    per_corpus = {}
    for corpus in a.corpus:
        if _embed_first:
            cmd_embed(argparse.Namespace(**{**vars(a), "corpus": [corpus]}))
        demb, qemb, cids, qids, qrels = _load_emb(a, corpus)
        rows = evaluate(demb, qemb, cids, qids, qrels, methods,
                        exclude_self=corpus in datasets.SELF_MATCH_CORPORA)
        per_corpus[corpus] = rows
        print(f"\n=== {a.embedder} x {corpus} ({demb.shape[1]}-d) ===")
        print(f"  {'method':16s}{'bytes':>8s}{'nDCG@10':>9s}{'recall@10':>11s}{'% float':>9s}")
        for r in rows:
            print(f"  {r['method']:16s}{r['bytes']:8.0f}{r['ndcg']:9.3f}{r['recall']:11.3f}{r['pct_float']:8.0f}%")
    os.makedirs(a.out, exist_ok=True)
    card = dict(embedder=a.embedder, corpora=a.corpus,
                dim=int(_load_emb(a, a.corpus[0])[0].shape[1]),
                per_corpus=per_corpus, aggregate=aggregate(per_corpus))
    path = os.path.join(a.out, f"card_{a.embedder}.json")
    json.dump(card, open(path, "w"), indent=2)
    print(f"\n=== aggregate over {len(a.corpus)} corpora (mean +/- std) ===")
    for r in card["aggregate"]:
        print(f"  {r['method']:16s}{r['bytes']:8.0f}B  nDCG {r['ndcg']:.3f}  "
              f"{r['pct_float']:.0f}+/-{r['pct_float_std']:.0f}% of float")
    print(f"\nwrote results card -> {path}")
    return card


def cmd_methods(a):
    print("registered methods (name | axis | bits):")
    for n in list_methods():
        fn = METHODS[n]
        print(f"  {n:16s} {getattr(fn,'axis','?'):12s} {fn.bits}")


def cmd_indexes(a):
    print("registered indexes (name | axis):")
    for n in list_indexes():
        print(f"  {n:14s} {getattr(INDEXES[n], 'axis', 'organisation')}")


def cmd_bench_index(a):
    """Build each index over document vectors and report recall@k, QPS and bytes/vec.

    faiss (its OpenMP) and the bit-trie (libomp for the C kernel) cannot share one process on
    macOS (two OpenMP runtimes). When both groups are requested we run each in its own subprocess
    and merge; the data is deterministic from the arguments, so the split is transparent.
    """
    sel = a.indexes or list_indexes()
    faiss_grp = [n for n in sel if getattr(INDEXES[n], "axis", "") == "organisation"]
    other_grp = [n for n in sel if n not in faiss_grp]
    if faiss_grp and other_grp and not getattr(a, "no_split", False):
        import subprocess
        import sys
        import tempfile
        import shutil
        merged, meta = [], {}
        for grp in (faiss_grp, other_grp):
            td = tempfile.mkdtemp()
            cmd = [sys.executable, "-m", "bitbudget.cli", "bench-index",
                   *_bench_data_args(a), "--k", str(a.k), "--out", td, "--no-split",
                   "--indexes", *grp]
            subprocess.run(cmd, check=True, env=os.environ.copy())
            c = json.load(open(os.path.join(td, "index_card.json")))
            merged += c["rows"]; meta = c
            shutil.rmtree(td, ignore_errors=True)
        merged.sort(key=lambda r: r["bytes"])
        os.makedirs(a.out, exist_ok=True)
        card = dict(source=meta["source"], k=a.k, dim=meta["dim"], n=meta["n"], rows=merged)
        json.dump(card, open(os.path.join(a.out, "index_card.json"), "w"), indent=2)
        print("\n=== board (faiss + bit-trie run in separate processes) ===")
        for r in merged:
            print(f"  {r['method']:14s}{r['bytes']:8.0f}B  recall@{a.k}={r['recall']:.3f}  {r['qps']:9.0f} qps")
        print(f"wrote index card -> {os.path.join(a.out, 'index_card.json')}")
        return card

    xb, xq, label = _bench_load(a)
    print(f"[bench-index] {label}: base={tuple(xb.shape)}, queries={len(xq)}, k={a.k}")
    rows = bench_indexes(xb, xq, k=a.k, which=sel)
    os.makedirs(a.out, exist_ok=True)
    card = dict(source=label, k=a.k, dim=int(xb.shape[1]), n=int(len(xb)), rows=rows)
    path = os.path.join(a.out, "index_card.json")
    json.dump(card, open(path, "w"), indent=2)
    print(f"\nwrote index card -> {path}")
    return card


def _bench_data_args(a):
    if a.synthetic:
        return ["--synthetic", str(a.synthetic[0]), str(a.synthetic[1])]
    if a.npz:
        return ["--npz", a.npz]
    return ["--embedder", a.embedder, "--corpus", a.corpus, "--cache", a.cache]


def _bench_load(a):
    if a.synthetic:
        N, D = a.synthetic
        rng = np.random.RandomState(0)
        centers = rng.randn(max(2, N // 500), D).astype(np.float32) * 4.0
        xb = centers[rng.randint(0, len(centers), N)] + rng.randn(N, D).astype(np.float32)
        nq = min(10000, max(1, N // 10))                  # enough queries to time throughput stably
        xq = xb[rng.choice(N, nq, replace=False)] + 0.1 * rng.randn(nq, D).astype(np.float32)
        return xb, xq, f"synthetic N={N} D={D}"
    if a.npz:
        z = np.load(os.path.expanduser(a.npz), allow_pickle=True)
        xb = z["base"] if "base" in z.files else z["demb"]
        xq = z["query"] if "query" in z.files else z["qemb"]
        return xb, xq, a.npz
    if a.embedder and a.corpus:
        z = np.load(_emb_path(a.cache, a.embedder, a.corpus), allow_pickle=True)
        return z["demb"], z["qemb"], f"{a.embedder} x {a.corpus}"
    raise SystemExit("bench-index needs one of: --synthetic N D | --npz PATH | --embedder X --corpus Y")


def cmd_embedders(a):
    from .embedders import list_embedders, EMBEDDERS
    print("registered embedders (name | dim | matryoshka):")
    for n in list_embedders():
        fn = EMBEDDERS[n]
        print(f"  {n:10s} dim={fn.dim}  matryoshka={fn.matryoshka}  ({fn.model})")


def cmd_leaderboard(a):
    cards = [json.load(open(p)) for pat in a.cards for p in glob.glob(pat)]
    print("# BitBudget leaderboard\n")
    for c in sorted(cards, key=lambda c: -c["dim"]):
        print(f"## {c['embedder']} ({c['dim']}-d), mean over {len(c['corpora'])} corpora\n")
        print("| Method | Axis | Bytes/vec | nDCG@10 | % of float |")
        print("|---|---|---|---|---|")
        for r in c["aggregate"]:
            print(f"| {r['method']} | {r['axis']} | {r['bytes']:.0f} | "
                  f"{r['ndcg']:.3f} | {r['pct_float']:.0f} ± {r['pct_float_std']:.0f} |")
        print()


def main(argv=None):
    p = argparse.ArgumentParser(prog="bitbudget", description="recall-per-byte benchmark for embedding compression")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("embed", "eval", "run"):
        s = sub.add_parser(name)
        s.add_argument("--embedder", required=True)
        s.add_argument("--corpus", nargs="+", default=datasets.CORPORA)
        s.add_argument("--methods", nargs="+", default=None)
        s.add_argument("--cache", default="~/.cache/bitbudget")
        s.add_argument("--out", default="results")
        s.add_argument("--device", default="cpu", help="cpu | mps (embedding step)")
        s.add_argument("--batch-size", type=int, default=64)
        s.add_argument("--split", default="test")
        s.add_argument("--force", action="store_true", help="recompute embeddings even if cached")
    sub.add_parser("methods")
    sub.add_parser("embedders")
    sub.add_parser("indexes")
    lb = sub.add_parser("leaderboard"); lb.add_argument("cards", nargs="+")
    bi = sub.add_parser("bench-index")
    bi.add_argument("--synthetic", nargs=2, type=int, metavar=("N", "D"),
                    help="benchmark on N clustered Gaussian vectors of dimension D")
    bi.add_argument("--npz", help="load base/query (keys base,query or demb,qemb) from an .npz")
    bi.add_argument("--embedder", help="use a cached embedding (with --corpus): demb as base")
    bi.add_argument("--corpus")
    bi.add_argument("--cache", default="~/.cache/bitbudget")
    bi.add_argument("--k", type=int, default=10)
    bi.add_argument("--indexes", nargs="+", default=None, help="subset of indexes to run")
    bi.add_argument("--out", default="results")
    bi.add_argument("--no-split", dest="no_split", action="store_true",
                    help="run all indexes in one process (do not isolate faiss from the bit-trie)")
    a = p.parse_args(argv)
    if a.cmd == "embed":   cmd_embed(a)
    elif a.cmd == "eval":  cmd_eval(a)
    elif a.cmd == "run":   cmd_eval(a, _embed_first=True)
    elif a.cmd == "methods":     cmd_methods(a)
    elif a.cmd == "embedders":   cmd_embedders(a)
    elif a.cmd == "indexes":     cmd_indexes(a)
    elif a.cmd == "leaderboard": cmd_leaderboard(a)
    elif a.cmd == "bench-index": cmd_bench_index(a)


if __name__ == "__main__":
    main()
