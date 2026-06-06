#!/usr/bin/env bash
# Regenerate every compression-leaderboard card: each embedder x the 4 BEIR corpora, scored by
# all registered methods. Embedding is cached, so re-runs are cheap. OpenAI embedders need
# OPENAI_API_KEY in the environment. Cards land in board_cards/; convert with cards_to_data.py.
set -u
cd "$(dirname "$0")"
P=/opt/homebrew/bin/python3.11
OUT=board_cards
mkdir -p "$OUT"
CORPORA="scifact nfcorpus arguana"
LOCAL="minilm mxbai bge-base e5-base gte-base"
HOSTED="openai-3-small openai-3-large"

for E in $LOCAL $HOSTED; do
  echo "=================== $E ==================="
  $P -m bitbudget.cli run --embedder "$E" --corpus $CORPORA --out "$OUT" 2>&1 \
     | grep -vE 'Batches|it/s|hf_xet|resolve' | tail -16 \
     || echo "  [warn] $E failed, continuing"
done
echo "DONE. cards:"
ls -1 "$OUT"/card_*.json
