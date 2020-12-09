#!/bin/bash

GRAPHS=$1
NODES=$2

cd /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain

declare -a types=("cipher" "hash" "tls" "iv" "key" "hnv" "hnvor" "tm")
# add only_labeled_type option for overrides

for type in "${types[@]}"; do
    python get_embed.py --data_path /Users/felixfischer/Documents/deep-learning-on-partial-programs/data/features/$type/insecure/ --iter_level 5 --load_path /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/saved_model_128_5_100/graphnn-model-100 --embed_dim 128 --output_dim 128 --type insecure_$type --nodes $GRAPHS
    python get_embed.py --data_path /Users/felixfischer/Documents/deep-learning-on-partial-programs/data/features/$type/secure/ --iter_level 5 --load_path /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/saved_model_128_5_100/graphnn-model-100 --embed_dim 128 --output_dim 128 --type secure_$type --nodes $GRAPHS
done
