# TCR Pretrain Technical Design

## Overview

TCR (Transparent Conflict Resolution) pretrain system with 3 stages.

## Signal Computation

### Stage 1 - Dual Encoder

Train semantic and factual projectors on Wikidata-Conflict-5K.

- σ_sem: semantic similarity (paraphrase, original) vs (unrelated, original)
- σ_fact: factual consistency (paraphrase, original) vs (contradiction, original)

### Stage 2 - Answerability MLP

Predict answerability from Llama middle-layer hidden states.

- Architecture: Linear(12288, 512) → ReLU → Dropout → Linear(512, 128) → Linear(128, 2)
- Extract layer: `int(num_layers * 2/3)`

### Stage 3 - End-to-End

Combine pretrained signals with trainable soft tokens.

## Files

| File | Description |
|------|-------------|
| `stage1_train_dual_encoder_v2.py` | Stage 1 training |
| `stage2_train_answerability_mlp.py` | Stage 2 training |
| `prepare.py` | Stage 3 data preparation |
| `train.py` | Stage 3 training |

## Output Checkpoints

- `checkpoints/dual_encoder_v2__Salesforce-SFR-Embedding-Mistral.pt`
- `checkpoints/answerability_mlp__meta-llama-Llama-3.1-8B-Instruct.pt`
