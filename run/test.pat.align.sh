#!/bin/bash

mtypes=("bert" "scibert" "biobert" "pubmedbert" "biobert-stsb")

cd .. || exit

for model in "${mtypes[@]}"; do
  poetry run python -m run.experiments.pattern_tensor_align \
    --model-type "$model" \
    "$@"
done
