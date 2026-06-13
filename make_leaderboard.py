#!/usr/bin/env python3
"""Render LEADERBOARD.md from site/data.json's compression board, so the markdown table
never drifts from the live site. Run after cards_to_data.py. One tab per embedder, rows in the
site's byte-sorted order; `% of float` is bolded for the lossless rows (the site's `lossless`
flag, pct >= 99.5)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "site", "data.json")
OUT = os.path.join(HERE, "LEADERBOARD.md")

HEADER = """\
# BitBudget leaderboard

<!-- Generated from site/data.json by make_leaderboard.py — do not edit by hand. -->

**Live leaderboard:** https://sjmoran.github.io/bitbudget/ (regenerated from `site/data.json` on merge)
**Paper:** https://arxiv.org/abs/2510.04127

{protocol}

**To submit:** register your method (`@method(...)`), run
`bitbudget run --embedder <name> --corpus scifact nfcorpus arguana`, and open a PR adding
your row with the generated results card attached. See [CONTRIBUTING.md](CONTRIBUTING.md).
"""

READING = """\
**Reading it.** At a matched footprint the quantisation-axis methods (binary, PQ, RaBitQ)
dominate the projection-axis ones (Matryoshka, PCA): bits beat dimensions. A one-bit code with a
re-ranking pass (`binary+rerank`) is essentially lossless at 32x compression on every embedder
except the strongly off-centre e5-base-v2, where a zero threshold misallocates its bits;
re-centring the threshold at each coordinate's mean (`binary-mean`) recovers it. `% of float` is
nDCG@10 relative to the uncompressed embedding; `±` is the standard deviation across the corpora.
"""


def pct_cell(row):
    if row["axis"] == "none":            # the float32 baseline is the 100% reference
        return "100"
    cell = f"{row['pct']:.1f} ± {row['std']:.1f}"
    return f"**{cell}**" if row.get("lossless") else cell


def main():
    d = json.load(open(DATA))
    out = [HEADER.format(protocol=d["protocol"])]
    for e in d["embedders"]:
        out.append(f"## {e['name']} ({e['note']})\n")
        out.append("| Method | Axis | Bytes/vec | nDCG@10 | % of float |")
        out.append("|---|---|---|---|---|")
        for r in e["rows"]:
            out.append(f"| {r['method']} | {r['axis']} | {r['bytes']} | "
                       f"{r['ndcg']:.3f} | {pct_cell(r)} |")
        out.append("")
    out.append(READING)
    open(OUT, "w").write("\n".join(out))
    print(f"wrote {OUT}: {len(d['embedders'])} embedder tabs")


if __name__ == "__main__":
    main()
