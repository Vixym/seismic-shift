#!/bin/bash
# Full recall_90->97 query sweep on the pruned full index.
set -u
IDX=/data/scratch/vihan/sparse_datasets/indexes/splade_full_pruned
Q=/data/scratch/vihan/sparse_datasets/data/queries.bin
OUTDIR=/tmp/sweep_pruned
mkdir -p "$OUTDIR"
BIN=./build/bin/perf_inverted_index

# subsection: query-cut heap-factor
declare -A CUT=( [recall_90]=3 [recall_91]=4 [recall_93]=5 [recall_94]=8 [recall_95]=5 [recall_97]=6 )
declare -A HF=(  [recall_90]=0.9 [recall_91]=0.9 [recall_93]=0.9 [recall_94]=0.9 [recall_95]=0.8 [recall_97]=0.7 )

for s in recall_90 recall_91 recall_93 recall_94 recall_95 recall_97; do
  echo "=== $s (cut ${CUT[$s]}, hf ${HF[$s]}) $(date +%T) ==="
  $BIN --index-file "$IDX" -k 10 --query-file "$Q" \
       --query-cut "${CUT[$s]}" --heap-factor "${HF[$s]}" --n-runs 1 \
       --output-path "$OUTDIR/$s" 2>/dev/null | grep -E "microsecs per query"
done
echo "=== SWEEP DONE $(date +%T) ==="
