"""BitBudget command line.

    bitbudget methods                              # list registered compression methods
    bitbudget embedders                            # list registered embedders
    bitbudget embed   --embedder mxbai --corpus scifact      # torch step -> cached embeddings
    bitbudget eval    --embedder mxbai --corpus scifact      # numpy step -> results card
    bitbudget run     --embedder mxbai --corpus scifact nfcorpus arguana fiqa   # embed + eval
    bitbudget leaderboard results/*.json           # aggregate cards into a markdown table

`run` embeds (torch) and evaluates (numpy) in one process, which is safe because the core
methods use no faiss. If you add a faiss-backed method, run `embed` and `eval` separately.
"""
import argparse
import glob
import json
import os

import numpy as np

from . import datasets
from .eval import evaluate, aggregate
from .methods import list_methods, METHODS


def _emb_path(cache, embedder, corpus):
    return os.path.join(os.path.expanduser(cache), f"emb_{embedder}_{corpus}.npz")


def cmd_embed(a):
    from .embedders import EMBEDDERS                      # lazy: imports torch
    emb = EMBEDDERS[a.embedder]
    for corpus in a.corpus:
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
        rows = evaluate(demb, qemb, cids, qids, qrels, methods)
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
    sub.add_parser("methods")
    sub.add_parser("embedders")
    lb = sub.add_parser("leaderboard"); lb.add_argument("cards", nargs="+")
    a = p.parse_args(argv)
    if a.cmd == "embed":   cmd_embed(a)
    elif a.cmd == "eval":  cmd_eval(a)
    elif a.cmd == "run":   cmd_eval(a, _embed_first=True)
    elif a.cmd == "methods":     cmd_methods(a)
    elif a.cmd == "embedders":   cmd_embedders(a)
    elif a.cmd == "leaderboard": cmd_leaderboard(a)


if __name__ == "__main__":
    main()
