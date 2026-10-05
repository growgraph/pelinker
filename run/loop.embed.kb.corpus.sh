#!/bin/bash
# Embed a corpus against the KB for every backbone x layer combination (fit stage A only),
# writing one mention parquet per combination for the hyperparameter searches.
#
# Existing parquets are never overwritten: a previous grid is evidence (for example the
# "before" arm of a weak-label change), so the run stops on the first file that exists.
# Write each grid to a new --output-parquet-path directory.
#
# Usage:
#   ./run/loop.embed.kb.corpus.sh --input-text-table-path PATH --output-parquet-path DIR \
#       --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
#       [--models "pubmedbert scibert"] [--layers "1 2"] [--max-input-buffers N] [--no-gpu]

set -euo pipefail

# Default grid
mtypes=("bert" "biobert" "pubmedbert" "bluebert" "scibert")
layers=("1" "2" "3")

INPUT_TEXT_TABLE_PATH=""
OUTPUT_PARQUET_PREFIX=""
KB_CSV_PATH=""
MAX_INPUT_BUFFERS=""
USE_GPU=1

usage() {
    echo "Usage: $0 --input-text-table-path PATH --output-parquet-path DIR --kb-csv-path PATH"
    echo "          [--models \"m1 m2\"] [--layers \"1 2\"] [--max-input-buffers N] [--no-gpu]"
}

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
        --models)
            read -r -a mtypes <<< "${2//,/ }"
            shift 2
            ;;
        --layers)
            read -r -a layers <<< "${2//,/ }"
            shift 2
            ;;
        --max-input-buffers)
            MAX_INPUT_BUFFERS="$2"
            shift 2
            ;;
        --no-gpu)
            USE_GPU=0
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

for required in INPUT_TEXT_TABLE_PATH OUTPUT_PARQUET_PREFIX KB_CSV_PATH; do
    if [[ -z "${!required}" ]]; then
        echo "Error: missing required option for ${required}"
        usage
        exit 1
    fi
done

# The pairs KB carries is_symmetric; without it symmetric relations can be given an
# inverse direction in the weak labels.
if ! head -n 1 "$KB_CSV_PATH" | tr -d '\r' | tr ',' '\n' | grep -qx "is_symmetric"; then
    echo "Warning: $KB_CSV_PATH has no is_symmetric column; pass the pairs KB" \
         "(data/derived/properties.synthesis.2.pairs.csv) for symmetric handling."
fi

mkdir -p "$OUTPUT_PARQUET_PREFIX"

# Refuse before doing any work, so a half-finished grid is never mixed with an old one.
for model in "${mtypes[@]}"; do
    for layer in "${layers[@]}"; do
        output_file="${OUTPUT_PARQUET_PREFIX}/res_${model}_${layer}.parquet"
        if [[ -e "$output_file" ]]; then
            echo "Error: $output_file already exists; choose a new --output-parquet-path"
            exit 1
        fi
    done
done

extra_args=()
if [[ -n "$MAX_INPUT_BUFFERS" ]]; then
    extra_args+=(--max-input-buffers "$MAX_INPUT_BUFFERS")
fi
if [[ "$USE_GPU" == "1" ]]; then
    extra_args+=(--use-gpu)
fi

for model in "${mtypes[@]}"; do
    for layer in "${layers[@]}"; do
        output_file="${OUTPUT_PARQUET_PREFIX}/res_${model}_${layer}.parquet"
        echo "== ${model} layer ${layer} -> ${output_file}"
        uv run python run/embed_kb_corpus.py \
               --input-text-table-path "$INPUT_TEXT_TABLE_PATH" \
               --output-parquet-path "$output_file" \
               --kb-csv-path "$KB_CSV_PATH" \
               --model-type "$model" \
               --nlp-model en_core_web_lg \
               --layers-spec "$layer" \
               --encoder-batch-size 100 \
               --input-buffer-rows 2000 \
               --negatives-per-positive 1.0 \
               --negative-seed 13 \
               ${extra_args[@]+"${extra_args[@]}"}
    done
done
