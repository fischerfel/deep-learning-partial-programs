#!/bin/bash

cd /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework

declare -a types_const=("cipher" "hash" "tls")
declare -a types_init=("iv" "key")
declare -a type_hnv=("hnv")
declare -a type_hnvor=("hnvor" "tm")

for type in "${types_const[@]}"; do
    python snippet_classifier.py --predict /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/results/128_vertices_output_insecure_$type.csv --embed_size 128
    python snippet_classifier.py --predict /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/results/128_vertices_output_secure_$type.csv --embed_size 128
done

for type in "${types_init[@]}"; do
    python snippet_classifier.py --predict /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/results/128_vertices_output_insecure_$type.csv --embed_size 128
    python snippet_classifier.py --predict /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/results/128_vertices_output_secure_$type.csv --embed_size 128
done

for type in "${types_hnv[@]}"; do
    python snippet_classifier.py --predict /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/results/128_vertices_output_insecure_$type.csv --embed_size 128
    python snippet_classifier.py --predict /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/results/128_vertices_output_secure_$type.csv --embed_size 128
done

for type in "${types_hnvor[@]}"; do
    python snippet_classifier.py --predict /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/results/128_vertices_output_insecure_$type.csv --embed_size 128
    python snippet_classifier.py --predict /Users/felixfischer/Documents/deep-learning-on-partial-programs/deep-learning-bug-search/retrain_framework/retrain/results/128_vertices_output_secure_$type.csv --embed_size 128
done


