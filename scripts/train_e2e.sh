#!/bin/bash
# Stage 3: TCR End-to-End Training

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default config
LLM_NAME="meta-llama/Llama-3.1-8B-Instruct"
DATA_FILE="tcr_e2e/data/tcr_training_data.jsonl"
NUM_EPOCHS=10
BATCH_SIZE=2
LR=1e-4
NUM_SOFT_TOKENS=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            SMOKE="--smoke"
            NUM_EPOCHS=2
            BATCH_SIZE=1
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
echo "Stage 3: TCR End-to-End Training"
echo "=============================================="
echo "  LLM: $LLM_NAME"
echo "  Data: $DATA_FILE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  LR: $LR"
echo "  Soft tokens: $NUM_SOFT_TOKENS"
echo "=============================================="

python train_e2e.py \
    --data_file "$DATA_FILE" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    $SMOKE

echo ""
echo "Training complete!"
echo "Checkpoint: tcr_e2e/outputs/tcr_e2e_checkpoint.pt"
