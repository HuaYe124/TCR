#!/bin/bash
# Prepare E2E training data with precomputed signals

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default config
N_SAMPLES=20
WIKIDATA_FILE="tcr_pretrain/data/wikidata_conflict_5k.jsonl"
OUTPUT_FILE="tcr_e2e/data/tcr_training_data.jsonl"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            SMOKE="--smoke"
            N_SAMPLES=5
            shift
            ;;
        --n)
            N_SAMPLES="$2"
            shift 2
            ;;
        --data_file)
            WIKIDATA_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "=============================================="
echo "Prepare E2E Training Data"
echo "=============================================="
echo "  Samples: $N_SAMPLES"
echo "  Wikidata: $WIKIDATA_FILE"
echo "  Output: $OUTPUT_FILE"
echo "=============================================="

python _prepare_data.py --n "$N_SAMPLES" $SMOKE

echo ""
echo "Data saved to: $OUTPUT_FILE"
