#!/bin/bash

mtypes=("bert" "scibert" "biobert" "pubmedbert" "biobert-stsb")
cd .. || exit

for p in "${mtypes[@]}"; do
  poetry run python -m run.experiments.pattern_tensor_align --model-type $p
done
