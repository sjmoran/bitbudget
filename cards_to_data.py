#!/usr/bin/env python3
"""Fold the measured result cards in board_cards/ into site/data.json's compression board.
Replaces the `embedders` array (one tab per embedder) and tops up the method glossary; leaves
the scale / indexbench / supervised sections untouched. Every number here is reproducible with
`bitbudget run --embedder <name> --corpus scifact nfcorpus arguana fiqa`."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS = os.path.join(HERE, "board_cards")
DATA = os.path.join(HERE, "site", "data.json")

# display name, short note, and tab order for each embedder key
META = {
    "openai-3-small": ("text-embedding-3-small", "1536-d · hosted · Matryoshka"),
    "openai-3-large": ("text-embedding-3-large", "3072-d · hosted · Matryoshka"),
    "mxbai":          ("mxbai-embed-large",      "1024-d · Matryoshka"),
    "bge-base":       ("bge-base-en-v1.5",       "768-d"),
    "gte-base":       ("gte-base",               "768-d"),
    "e5-base":        ("e5-base-v2",             "768-d"),
    "minilm":         ("all-MiniLM-L6-v2",       "384-d"),
}
ORDER = ["openai-3-small", "openai-3-large", "mxbai", "bge-base", "gte-base", "e5-base", "minilm"]

# glossary entries for the methods added alongside the new embedders
NEW_METHOD_DESCS = {
    "float16": ("quantisation", "Half precision: each coordinate as a 16-bit float (2x smaller). "
                "The cheapest, most widely deployed compression, and essentially lossless."),
    "int4":    ("quantisation", "4-bit per-dimension scalar quantisation (15 signed levels): half "
                "the footprint of int8, an aggressive-but-not-binary operating point."),
    "binary-mean": ("quantisation", "1-bit sign code thresholded at the per-coordinate MEAN, not "
                "at zero. A zero threshold assumes mean-centred data (LSH assumption A1); for "
                "off-centre embeddings like e5 it wastes most bits, and re-centring the threshold "
                "recovers them (e5: 53% -> 86% of float). Where the threshold is placed can matter "
                "as much as how many bits are spent."),
}


def to_rows(aggregate):
    rows = []
    for a in aggregate:
        b = a["bytes"]
        rows.append({
            "method": a["method"],
            "axis": a["axis"],
            "bytes": int(b) if float(b).is_integer() else round(b, 1),
            "ndcg": round(a["ndcg"], 3),
            "pct": round(a["pct_float"], 1),
            "std": round(a.get("pct_float_std", 0.0), 1),
            "lossless": a["pct_float"] >= 99.5,
        })
    rows.sort(key=lambda r: r["bytes"])
    return rows


def main():
    cards = {}
    for path in glob.glob(os.path.join(CARDS, "card_*.json")):
        c = json.load(open(path))
        cards[c["embedder"]] = c

    embedders = []
    for key in ORDER:
        if key not in cards:
            print(f"[skip] no card for {key}")
            continue
        name, note = META[key]
        c = cards[key]
        embedders.append({"name": name, "dim": int(c["dim"]), "note": note,
                          "rows": to_rows(c["aggregate"])})
        print(f"[ok] {key:16s} -> {name} ({c['dim']}-d, {len(c['aggregate'])} methods)")

    d = json.load(open(DATA))
    d["embedders"] = embedders
    for m, (axis, desc) in NEW_METHOD_DESCS.items():
        d.setdefault("methods", {})[m] = {"axis": axis, "desc": desc}
    json.dump(d, open(DATA, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {DATA}: {len(embedders)} embedder tabs, {len(d['methods'])} methods in glossary")


if __name__ == "__main__":
    main()
