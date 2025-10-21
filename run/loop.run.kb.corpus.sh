#!/bin/bash

mtypes=("bert" "biobert" "pubmedbert" "bluebert" "scibert")

layers=("1" "2" "3")

for model in "${mtypes[@]}"; do
    for layer in "${layers[@]}"; do
	python run/run_kb_corpus.py --input-text-table-path data/jamshid/bio_mag_100K.tsv.gz \
	       --output-parquet-path data/jamshid/res_${model}_${layer}.parquet \
	       --properties-txt-path data/jamshid/verbial_props.txt \
	       --model-type $model \
	       --nlp-model en_core_web_trf \
	       --layers-spec $layer \
	       --batch-size 100 \
	       --chunk-size 2000 \
	       --use-gpu
    done
done
