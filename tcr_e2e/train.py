"""
TCR End-to-End Training.

Process:
1. Load pretrained Stage1/Stage2 encoders
2. Load training data (with σ_sem, σ_fact, σ_ans)
3. Train SignalProjector
4. Batch padding handling
5. F1 evaluation

Usage:
    python train.py --smoke
    python train.py --epochs 10
"""

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
if os.environ.get("HTTPS_PROXY"):
    pass  # use system proxy
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
DATA_FILE = PROJECT_ROOT / "data" / "tcr_training_data.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import argparse
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from model import TCRModel
from utils import evaluate_f1

print("=" * 60)
print("TCR End-to-End Training")
print("=" * 60)
print(f"Device: {DEVICE}")


class TCRDataset(Dataset):
    """TCR training dataset."""

    def __init__(self, data_file):
        self.samples = []

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.samples.append(rec)

        print(f"Loaded {len(self.samples)} samples from {data_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return {
            "input_ids": torch.tensor(sample["prompt_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["prompt_mask"], dtype=torch.long),
            "prompt_len": sample["prompt_len"],
            "sigma_sem": torch.tensor(sample["sigma_sem"], dtype=torch.float32),
            "sigma_fact": torch.tensor(sample["sigma_fact"], dtype=torch.float32),
            "sigma_ans": torch.tensor(sample["sigma_ans"], dtype=torch.float32),
            "question": sample["question"],
            "answer": sample["golden_answer"],
            "context_type": sample.get("context_type", "golden"),
        }


def collate_fn(batch):
    """Collate batch with padding."""
    max_len = max(len(s["input_ids"]) for s in batch)

    padded_ids = []
    padded_labels = []
    padded_masks = []
    prompt_lens = []
    sigma_sems = []
    sigma_facts = []
    sigma_anss = []

    for s in batch:
        seq_len = len(s["input_ids"])
        pad_len = max_len - seq_len

        if pad_len > 0:
            ids = torch.cat([s["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
            lbls = torch.cat([s["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            masks = torch.cat([s["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        else:
            ids = s["input_ids"]
            lbls = s["labels"]
            masks = s["attention_mask"]

        padded_ids.append(ids)
        padded_labels.append(lbls)
        padded_masks.append(masks)
        prompt_lens.append(s["prompt_len"])
        sigma_sems.append(s["sigma_sem"])
        sigma_facts.append(s["sigma_fact"])
        sigma_anss.append(s["sigma_ans"])

    return {
        "input_ids": torch.stack(padded_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(padded_masks),
        "prompt_lens": prompt_lens,
        "sigma_sem": torch.stack(sigma_sems),
        "sigma_fact": torch.stack(sigma_facts),
        "sigma_ans": torch.stack(sigma_anss),
        "questions": [s["question"] for s in batch],
        "answers": [s["answer"] for s in batch],
        "context_types": [s.get("context_type", "golden") for s in batch],
    }


def train(args):
    """Training function."""

    print("\n[1] Loading data...")
    dataset = TCRDataset(DATA_FILE)

    if args.max_samples is not None:
        original_size = len(dataset)
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"  [LIMITED] Samples: {original_size} -> {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print("\n[2] Loading model...")

    print("  - Loading Llama...")
    llama_tokenizer = AutoTokenizer.from_pretrained(
        LLM_NAME, token=os.environ["HF_TOKEN"], trust_remote_code=True
    )
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, token=os.environ["HF_TOKEN"],
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    llama.eval()

    embed_dim = llama.config.hidden_size
    print(f"    Llama embed_dim: {embed_dim}")

    model = TCRModel(llama, embed_dim, num_soft_tokens=args.num_soft_tokens)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr, weight_decay=0.01)

    print("\n" + "=" * 60)
    print("[Pre-Training] F1 Evaluation")
    print("=" * 60)

    print("\nCalculating pre-training F1...")
    pre_f1, _, _ = evaluate_f1(model, dataloader, llama_tokenizer, DEVICE, num_samples=200)
    print(f"[Pre-Training] F1: {pre_f1:.4f}")

    print("\n[3] Training...")
    print(f"    Epochs: {args.epochs}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Steps per epoch: {len(dataloader)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            sigma_sem = batch["sigma_sem"].to(DEVICE)
            sigma_fact = batch["sigma_fact"].to(DEVICE)
            sigma_ans = batch["sigma_ans"].to(DEVICE)

            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                sigma_sem=sigma_sem,
                sigma_fact=sigma_fact,
                sigma_ans=sigma_ans,
                device=DEVICE,
                tokenizer=llama_tokenizer,
            )

            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"    Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\n  Epoch {epoch+1}/{args.epochs}: loss = {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = OUTPUT_DIR / f"tcr_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "signal_projector": model.signal_projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"    Saved: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt_path = OUTPUT_DIR / "tcr_best.pt"
            torch.save({
                "epoch": epoch,
                "signal_projector": model.signal_projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
            }, best_ckpt_path)
            print(f"    [Best] Saved: {best_ckpt_path}")

    print("\n" + "=" * 60)
    print("[Post-Training] F1 Evaluation")
    print("=" * 60)

    print("\nCalculating post-training F1...")
    post_f1, _, _ = evaluate_f1(model, dataloader, llama_tokenizer, DEVICE, num_samples=200)
    print(f"[Post-Training] F1: {post_f1:.4f}")

    print(f"\n[Summary] F1 Improvement: {pre_f1:.4f} -> {post_f1:.4f} ({post_f1 - pre_f1:+.4f})")

    final_path = OUTPUT_DIR / "tcr_final.pt"
    torch.save({
        "epoch": args.epochs,
        "signal_projector": model.signal_projector.state_dict(),
    }, final_path)
    print(f"\n[OK] Final model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_soft_tokens", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    if not os.environ.get("HF_TOKEN"):
        print("Error: Please set HF_TOKEN via environment or --hf_token")
        sys.exit(1)

    if args.smoke:
        args.epochs = 3
        args.batch_size = 2
        print("[SMOKE MODE]")

    train(args)


if __name__ == "__main__":
    main()
