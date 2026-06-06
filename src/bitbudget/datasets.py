"""BEIR corpus loading. Datasets auto-download to the cache dir on first use."""
import json
import os
import shutil
import urllib.request
import zipfile

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
# the standard small/medium BEIR corpora the leaderboard is defined over
CORPORA = ["scifact", "nfcorpus", "arguana", "fiqa"]


def _download(name, cache):
    d = os.path.join(cache, name)
    if not os.path.isdir(d):
        os.makedirs(cache, exist_ok=True)
        z = os.path.join(cache, name + ".zip")
        if not os.path.exists(z):
            req = urllib.request.Request(BEIR_URL.format(name), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, open(z + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(z + ".part", z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(cache)
    return d


def load(name, cache="~/.cache/bitbudget", split="test"):
    cache = os.path.expanduser(cache)
    d = _download(name, cache)
    corpus = {}
    with open(os.path.join(d, "corpus.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            corpus[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()
    queries = {}
    with open(os.path.join(d, "queries.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            queries[o["_id"]] = o["text"]
    qrels = {}
    with open(os.path.join(d, "qrels", split + ".tsv")) as f:
        next(f)
        for line in f:
            qid, did, score = line.strip().split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    queries = {q: queries[q] for q in qrels if q in queries}
    return corpus, queries, qrels
