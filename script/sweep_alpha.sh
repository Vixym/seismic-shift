#!/bin/bash
# Sweep the centroid+radius pruning strength alpha at fixed query params (recall_90).
# alpha=0 => legacy count-based centroid skip (baseline).
set -u
IDX=/data/scratch/vihan/sparse_datasets/indexes/splade_full_centroid_radius
Q=/data/scratch/vihan/sparse_datasets/data/queries.bin
OUTDIR=/tmp/sweep_alpha
mkdir -p "$OUTDIR"
BIN=./build/bin/perf_inverted_index
CUT=3; HF=0.9

for a in 0 0.1 0.15 0.2 0.25 0.3 0.5 0.8 1.2; do
  echo "=== alpha=$a (cut $CUT, hf $HF) $(date +%T) ==="
  $BIN --index-file "$IDX" -k 10 --query-file "$Q" \
       --query-cut "$CUT" --heap-factor "$HF" --n-runs 1 --alpha "$a" \
       --output-path "$OUTDIR/a_${a}" 2>/dev/null | grep -E "microsecs per query|Docs evaluated" | tail -3
done
echo "=== ALPHA SWEEP DONE $(date +%T) ==="
