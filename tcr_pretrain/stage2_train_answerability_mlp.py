"""Stage 2: Train Answerability MLP on extracted Llama middle-layer hidden states.

Usage:
    python -m tcr_pretrain.stage2_train_answerability_mlp --smoke
    python -m tcr_pretrain.stage2_train_answerability_mlp

Output: checkpoints/answerability_mlp__meta-llama-Llama-3.1-8B-Instruct.pt
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
DATA_FILE = DATA_DIR / "qa_answerability_dataset.jsonl"
HIDDEN_DIR = DATA_DIR / "hidden_states"
CKPT_DIR = SCRIPT_DIR.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True, parents=True)

LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HIDDEN_DIM = 4096
LR = 1e-4
NUM_EPOCHS = 15
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR.mkdir(exist_ok=True, parents=True)
HIDDEN_DIR.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "train_answerability_mlp.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class QAAnswerabilityDataset(Dataset):
    def __init__(self, jsonl_path: str, split: str = "train"):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("split", "train") == split:
                        path = Path(rec["hidden_state_path"])
                        if path.exists():
                            self.records.append(rec)
                        else:
                            logger.warning(f"Missing hidden state: {path}")
                except (json.JSONDecodeError, KeyError):
                    pass
        logger.info(f"Loaded {len(self.records)} {split} records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        h = torch.load(Path(rec["hidden_state_path"]), map_location="cpu").float()
        label = int(rec["label"])
        return h, label


class AnswerabilityMLP(nn.Module):
    def __init__(self, hidden_dim: int = 12288, intermediate_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(intermediate_dim, 128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        logits = self.fc3(x)
        probs = F.softmax(logits, dim=-1)[:, 1]
        return logits, probs


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def compute_macro_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    p0 = (preds == 0).sum().item()
    n0 = (labels == 0).sum().item()
    p1 = (preds == 1).sum().item()
    n1 = (labels == 1).sum().item()

    tp0 = ((preds == 0) & (labels == 0)).sum().item()
    tp1 = ((preds == 1) & (labels == 1)).sum().item()

    prec0 = tp0 / max(p0, 1)
    rec0 = tp0 / max(n0, 1)
    f10 = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-8)

    prec1 = tp1 / max(p1, 1)
    rec1 = tp1 / max(n1, 1)
    f11 = 2 * prec1 * rec1 / max(prec1 + rec1, 1e-8)

    return (f10 + f11) / 2


def smoke_validate_mlp(mlp: AnswerabilityMLP, device: torch.device, d_llm: int):
    x = torch.randn(2, d_llm)
    logits, probs = mlp(x.to(device))
    logger.info(f"  MLP input shape: {x.shape}")
    logger.info(f"  MLP logits shape: {logits.shape}")
    logger.info(f"  MLP probs shape: {probs.shape}")
    logger.info(f"  Sample probs: {probs.detach().cpu().numpy()}")
    assert logits.shape == (2, 2), f"Wrong logits shape: {logits.shape}"
    assert not logits.isnan().any(), "MLP output has NaN"
    logger.info("  MLP smoke check PASSED")


def train(data_file: str = DATA_FILE, llm_name: str = LLM_NAME, hidden_dim: int = HIDDEN_DIM, lr: float = LR, num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE, device: torch.device = None, smoke: bool = False):
    if device is None:
        device = torch.device(DEVICE)

    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        raise FileNotFoundError(data_file)

    if smoke:
        num_epochs = 2
        batch_size = min(batch_size, 16)
        logger.info(f"[SMOKE] {num_epochs} epochs, batch_size={batch_size}")

    all_ds = QAAnswerabilityDataset(data_file, split="train")
    val_ds = QAAnswerabilityDataset(data_file, split="val")

    if len(val_ds) == 0:
        logger.info("No val split found. Using 80/20 random split.")
        indices = list(range(len(all_ds)))
        random.seed(42)
        random.shuffle(indices)
        val_size = max(1, int(len(indices) * 0.2))
        val_indices = set(indices[:val_size])
        train_indices = set(indices[val_size:])

        train_records = [all_ds.records[i] for i in train_indices]
        val_records = [all_ds.records[i] for i in val_indices]

        class SplitDataset(Dataset):
            def __init__(self, records):
                self.records = records
            def __len__(self):
                return len(self.records)
            def __getitem__(self, idx):
                rec = self.records[idx]
                h = torch.load(Path(rec["hidden_state_path"]), map_location="cpu").float()
                label = int(rec["label"])
                return h, label

        train_ds = SplitDataset(train_records)
        val_ds = SplitDataset(val_records)
        logger.info(f"Manual split: Train={len(train_ds)}, Val={len(val_ds)}")

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    sample_h, _ = train_ds[0]
    actual_dim = sample_h.shape[0]
    if actual_dim != hidden_dim:
        logger.info(f"Auto-detected hidden_dim={actual_dim} from data (config={hidden_dim})")
        hidden_dim = actual_dim

    mlp = AnswerabilityMLP(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    logger.info(f"Trainable parameters: {sum(p.numel() for p in mlp.parameters()) / 1e3:.1f}K")

    logger.info("=== Pre-training MLP validation ===")
    smoke_validate_mlp(mlp, device, hidden_dim)

    train_labels = [l for _, l in train_ds]
    pos_rate = sum(train_labels) / max(len(train_labels), 1)
    logger.info(f"  Train positive rate: {pos_rate:.3f}")
    if pos_rate < 0.1 or pos_rate > 0.9:
        logger.error("LABEL IMBALANCE: Positive rate is extreme!")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    best_val_acc = 0.0
    best_state = None
    train_history = []

    for epoch in range(num_epochs):
        mlp.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for hidden_states, labels in pbar:
            hidden_states = hidden_states.to(device)
            labels = labels.to(device).long()

            logits, _ = mlp(hidden_states)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(n_batches, 1)

        mlp.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for hidden_states, labels in val_dl:
                hidden_states = hidden_states.to(device)
                _, probs = mlp(hidden_states)
                preds = (probs > 0.5).long()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        if not all_preds:
            logger.warning("  Validation set is empty, skipping.")
            val_acc = 0.0
            val_f1 = 0.0
        else:
            all_preds = torch.cat(all_preds).flatten()
            all_labels = torch.cat(all_labels).flatten()
            val_acc = compute_accuracy(all_preds, all_labels)
            val_f1 = compute_macro_f1(all_preds, all_labels)

        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Train Loss={avg_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            logger.info(f"  New best val_acc={best_val_acc:.4f}")

        train_history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_accuracy": val_acc, "val_f1": val_f1})

        if math.isnan(avg_loss):
            logger.error("Loss is NaN! Stopping training.")
            break

    if best_state is not None:
        mlp.load_state_dict(best_state)
        mlp.to(device)
        logger.info(f"Restored best model (val_acc={best_val_acc:.4f})")

    return mlp, best_val_acc, val_f1, train_history


def save_checkpoint(mlp: AnswerabilityMLP, llm_name: str, hidden_dim: int, extract_layer: int, val_accuracy: float, val_f1: float, num_epochs: int, train_history: list = None):
    ckpt_name = llm_name.replace("/", "-")
    save_path = CKPT_DIR / f"answerability_mlp__{ckpt_name}.pt"

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(llm_name, local_files_only=True)
    num_layers = getattr(config, "num_hidden_layers", 32)
    actual_extract_layer = int(num_layers * (2 / 3))

    state = {
        "mlp_state_dict": mlp.state_dict(),
        "llm_name": llm_name,
        "llm_hidden_size": hidden_dim,
        "llm_extract_layer": actual_extract_layer,
        "num_layers": num_layers,
        "epoch": num_epochs,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "train_history": train_history or [],
    }
    torch.save(state, save_path)
    logger.info(f"Checkpoint saved: {save_path}")

    loaded = torch.load(save_path, map_location="cpu")
    assert "mlp_state_dict" in loaded, "Checkpoint missing mlp_state_dict"
    assert loaded["llm_name"] == llm_name, "llm_name mismatch"
    assert loaded["llm_hidden_size"] == hidden_dim, "hidden_size mismatch"
    logger.info("Checkpoint verification PASSED")


def main():
    parser = argparse.ArgumentParser(description="Train answerability MLP")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (2 epochs)")
    parser.add_argument("--data_file", type=str, default=str(DATA_FILE))
    parser.add_argument("--model_name", type=str, default=LLM_NAME)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stage 2: Answerability MLP Training")
    logger.info(f"  Data: {args.data_file}")
    logger.info(f"  LLM: {args.model_name}")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 60)

    mlp, val_acc, val_f1, train_history = train(
        data_file=args.data_file,
        llm_name=args.model_name,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        smoke=args.smoke,
    )

    logger.info("=" * 60)
    logger.info("Training History:")
    logger.info(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Acc':>8} | {'Val F1':>8}")
    logger.info("-" * 40)
    for h in train_history:
        logger.info(f"{h['epoch']:>6} | {h['train_loss']:>10.4f} | {h['val_accuracy']:>8.4f} | {h['val_f1']:>8.4f}")
    logger.info("=" * 60)

    save_checkpoint(
        mlp=mlp,
        llm_name=args.model_name,
        hidden_dim=args.hidden_dim,
        extract_layer=int(32 * 2 / 3),
        val_accuracy=val_acc,
        val_f1=val_f1,
        num_epochs=args.num_epochs,
        train_history=train_history,
    )
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
