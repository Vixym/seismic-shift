#!/bin/bash
# recall_90->97 sweep on the two dynamic indexes (max+none vs centroid+jlt).
set -u
IDXDIR=/data/scratch/vihan/sparse_datasets/indexes
Q=/data/scratch/vihan/sparse_datasets/data/queries.bin
OUTDIR=/tmp/sweep_dyn
mkdir -p "$OUTDIR"
BIN=./build/bin/perf_inverted_index

declare -A CUT=( [recall_90]=3 [recall_91]=4 [recall_93]=5 [recall_94]=8 [recall_95]=5 [recall_97]=6 )
declare -A HF=(  [recall_90]=0.9 [recall_91]=0.9 [recall_93]=0.9 [recall_94]=0.9 [recall_95]=0.8 [recall_97]=0.7 )

for cfg in splade_full_dyn_max_none splade_full_dyn_centroid_jlt; do
  for s in recall_90 recall_91 recall_93 recall_94 recall_95 recall_97; do
    echo "=== $cfg / $s (cut ${CUT[$s]}, hf ${HF[$s]}) $(date +%T) ==="
    $BIN --index-file "$IDXDIR/$cfg" -k 10 --query-file "$Q" \
         --query-cut "${CUT[$s]}" --heap-factor "${HF[$s]}" --n-runs 1 \
         --output-path "$OUTDIR/${cfg}__${s}" 2>/dev/null | grep -E "microsecs per query"
  done
done
echo "=== SWEEP DONE $(date +%T) ==="
