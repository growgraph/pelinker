#!/usr/bin/env bash
#
# Batch end-to-end linker training: same (model_type × layers) grid as loop.embed.kb.corpus.sh,
# but runs pelinker-fit (embed_kb_corpus stage A + Linker.fit stage B) per combination.
#
# From the repository root:
#   ./run/loop.fit.sh \
#     --input-text-table-path data/corpus/articles.tsv.gz \
#     --kb-csv-path data/derived/properties.synthesis.1.csv \
#     --output-parquet-prefix outputs/embeddings \
#     --output-model-prefix models/fit_runs \
#     [--layers 1,2,3]

set -euo pipefail

mtypes=("bert" "biobert" "pubmedbert" "bluebert" "scibert")
# Layer specs for pelinker-fit (each becomes layers_spec=...).
# - Comma-separated: "1,2,3" → three runs (1, 2, 3).
# - Use semicolons to pass comma-containing specs: "1,2;3" → two runs (1,2 and 3).
LAYERS_RAW="1,2,3"

INPUT_TEXT_TABLE_PATH=""
OUTPUT_PARQUET_PREFIX=""
OUTPUT_MODEL_PREFIX=""
KB_CSV_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-text-table-path)
            INPUT_TEXT_TABLE_PATH="$2"
            shift 2
            ;;
        --output-parquet-prefix)
            OUTPUT_PARQUET_PREFIX="$2"
            shift 2
            ;;
        --output-model-prefix)
            OUTPUT_MODEL_PREFIX="$2"
            shift 2
            ;;
        --kb-csv-path)
            KB_CSV_PATH="$2"
            shift 2
            ;;
        --layers)
            LAYERS_RAW="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --input-text-table-path PATH --kb-csv-path PATH \\"
            echo "          --output-parquet-prefix DIR --output-model-prefix DIR \\"
            echo "          [--layers SPECS]"
            echo "  --layers: layer specs (default: 1,2,3). Comma = separate specs;"
            echo "            use semicolons for one spec that contains commas, e.g. 1,2;3"
            exit 1
            ;;
    esac
done

if [[ "$LAYERS_RAW" == *';'* ]]; then
    IFS=';' read -r -a layers <<< "$LAYERS_RAW"
else
    IFS=',' read -r -a layers <<< "$LAYERS_RAW"
fi
if [[ ${#layers[@]} -eq 0 ]] || [[ -z "${layers[0]}" ]]; then
    echo "Error: --layers must list at least one layers_spec"
    exit 1
fi

if [[ -z "$INPUT_TEXT_TABLE_PATH" ]]; then
    echo "Error: --input-text-table-path is required"
    exit 1
fi

if [[ -z "$OUTPUT_PARQUET_PREFIX" ]]; then
    echo "Error: --output-parquet-prefix is required"
    exit 1
fi

if [[ -z "$OUTPUT_MODEL_PREFIX" ]]; then
    echo "Error: --output-model-prefix is required"
    exit 1
fi

if [[ -z "$KB_CSV_PATH" ]]; then
    echo "Error: --kb-csv-path is required"
    exit 1
fi

mkdir -p "$OUTPUT_PARQUET_PREFIX" "$OUTPUT_MODEL_PREFIX"

for model in "${mtypes[@]}"; do
    for layer in "${layers[@]}"; do
        layer_trimmed="${layer#"${layer%%[![:space:]]*}"}"
        layer_trimmed="${layer_trimmed%"${layer_trimmed##*[![:space:]]}"}"
        layer_tag="${layer_trimmed//,/_}"
        emb_file="${OUTPUT_PARQUET_PREFIX}/res_${model}_${layer_tag}.parquet"
        model_file="${OUTPUT_MODEL_PREFIX}/pelinker.${model}.${layer_tag}"
        uv run pelinker-fit \
            kb_path="$KB_CSV_PATH" \
            input_text_table_path="$INPUT_TEXT_TABLE_PATH" \
            embeddings_parquet="$emb_file" \
            output_path="$model_file" \
            model_type="$model" \
            layers_spec="$layer_trimmed" \
            nlp_model=en_core_web_trf \
            encoder_batch_size=100 \
            input_buffer_rows=2000 \
            use_gpu=true
    done
done
