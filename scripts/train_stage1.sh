#!/bin/bash
# Stage 1: Train Dual Encoder (semantic + factual projectors)

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default config
ENCODER_NAME="Salesforce/SFR-Embedding-Mistral"
DATA_FILE="tcr_pretrain/data/wikidata_conflict_5k.jsonl"
BATCH_SIZE=64
LR=1e-4
NUM_EPOCHS=5
TAU=0.07

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            SMOKE="--smoke"
            NUM_EPOCHS=2
            shift
            ;;
        --data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Stage 1: Dual Encoder Training"
echo "=============================================="
echo "  Encoder: $ENCODER_NAME"
echo "  Data: $DATA_FILE"
echo "  Batch size: $BATCH_SIZE"
echo "  LR: $LR"
echo "  Epochs: $NUM_EPOCHS"
echo "=============================================="

python -m tcr_pretrain.stage1_train_dual_encoder_v2 \
    --data_file "$DATA_FILE" \
    --encoder_name "$ENCODER_NAME" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --tau "$TAU" \
    $SMOKE

echo ""
echo "Checkpoint saved to: checkpoints/dual_encoder_v2__Salesforce-SFR-Embedding-Mistral.pt"
