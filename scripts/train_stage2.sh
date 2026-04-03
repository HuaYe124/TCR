#!/bin/bash
# Stage 2: Train Answerability MLP

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default config
LLM_NAME="meta-llama/Llama-3.1-8B-Instruct"
DATA_FILE="tcr_pretrain/data/qa_answerability_dataset.jsonl"
HIDDEN_DIM=4096
BATCH_SIZE=64
LR=1e-4
NUM_EPOCHS=15

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            SMOKE="--smoke"
            NUM_EPOCHS=2
            BATCH_SIZE=16
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
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Stage 2: Answerability MLP Training"
echo "=============================================="
echo "  LLM: $LLM_NAME"
echo "  Data: $DATA_FILE"
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Batch size: $BATCH_SIZE"
echo "  LR: $LR"
echo "  Epochs: $NUM_EPOCHS"
echo "=============================================="

python -m tcr_pretrain.stage2_train_answerability_mlp \
    --data_file "$DATA_FILE" \
    --model_name "$LLM_NAME" \
    --hidden_dim "$HIDDEN_DIM" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    $SMOKE

echo ""
echo "Checkpoint saved to: checkpoints/answerability_mlp__meta-llama-Llama-3.1-8B-Instruct.pt"
