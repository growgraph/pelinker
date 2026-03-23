#!/bin/bash

# Default values
mtypes=("bert" "biobert" "pubmedbert" "bluebert" "scibert")
layers=("1" "2" "3")

# Parse command-line arguments
INPUT_TEXT_TABLE_PATH=""
OUTPUT_PARQUET_PREFIX=""
KB_CSV_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-text-table-path)
            INPUT_TEXT_TABLE_PATH="$2"
            shift 2
            ;;
        --output-parquet-path)
            OUTPUT_PARQUET_PREFIX="$2"
            shift 2
            ;;
        --kb-csv-path)
            KB_CSV_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --input-text-table-path PATH --output-parquet-path PREFIX --kb-csv-path PATH"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$INPUT_TEXT_TABLE_PATH" ]]; then
    echo "Error: --input-text-table-path is required"
    exit 1
fi

if [[ -z "$OUTPUT_PARQUET_PREFIX" ]]; then
    echo "Error: --output-parquet-path is required"
    exit 1
fi

if [[ -z "$KB_CSV_PATH" ]]; then
    echo "Error: --kb-csv-path is required"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_PARQUET_PREFIX"

for model in "${mtypes[@]}"; do
    for layer in "${layers[@]}"; do
        output_file="${OUTPUT_PARQUET_PREFIX}/res_${model}_${layer}.parquet"
        python run/embed_kb_corpus.py \
               --input-text-table-path "$INPUT_TEXT_TABLE_PATH" \
               --output-parquet-path "$output_file" \
               --kb-csv-path "$KB_CSV_PATH" \
               --model-type "$model" \
               --nlp-model en_core_web_trf \
               --layers-spec "$layer" \
               --batch-size 100 \
               --chunk-size 2000 \
               --use-gpu
    done
done
