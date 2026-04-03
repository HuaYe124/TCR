"""Stage 1: Train Dual Encoder (semantic + factual projectors) on Wikidata-Conflict-5K.

Usage:
    python -m tcr_pretrain.stage1_train_dual_encoder_v2 --smoke
    python -m tcr_pretrain.stage1_train_dual_encoder_v2

Output: checkpoints/dual_encoder_v2__Salesforce-SFR-Embedding-Mistral.pt
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
if os.environ.get("HTTPS_PROXY"):
    os.environ.setdefault("HTTPS_PROXY", os.environ["HTTPS_PROXY"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
DATA_FILE = DATA_DIR / "wikidata_conflict_5k.jsonl"
CKPT_DIR = SCRIPT_DIR.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

SENTENCE_ENCODER_NAME = "Salesforce/SFR-Embedding-Mistral"
TAU = 0.07
BATCH_SIZE = 64
LR = 1e-4
NUM_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "train_dual_encoder_v2.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class WikidataConflictDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        logger.info(f"Loaded {len(self.records)} records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.records[idx]


def collate_conflict(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    return {
        "originals": [b["original"] for b in batch],
        "paraphrases": [b["paraphrase"] for b in batch],
        "contradictions": [b["contradiction"] for b in batch],
        "unrelateds": [b["unrelated"] for b in batch],
    }


class DualEncoderProjector(nn.Module):
    def __init__(self, hidden_dim: int, sub_dim: Optional[int] = None):
        super().__init__()
        if sub_dim is None:
            sub_dim = hidden_dim
        self.sub_dim = sub_dim
        self.encoder_sem = nn.Linear(hidden_dim, sub_dim)
        self.encoder_fact = nn.Linear(hidden_dim, sub_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder_sem(z), self.encoder_fact(z)


def cosine_similarity_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)
    return (z1_norm * z2_norm).sum(dim=1)


def load_sentence_encoder(name: str, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"Loading sentence encoder: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    encoder = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16)
    encoder.eval()
    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    embed_dim = encoder.config.hidden_size
    logger.info(f"Encoder hidden dim: {embed_dim}, device: {device}")
    return encoder, tokenizer, embed_dim


def encode_batch(encoder: nn.Module, tokenizer, texts: List[str], device: torch.device, max_length: int = 512) -> torch.Tensor:
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(hidden.size(0), device=device)
    return hidden[batch_idx, seq_lengths].float()


def infoNCE_loss(pos_sim: torch.Tensor, neg_sim: List[torch.Tensor], tau: float = 0.07) -> torch.Tensor:
    batch_size = pos_sim.size(0)
    pos_sim_scaled = pos_sim / tau
    pos_exp = torch.exp(pos_sim_scaled)
    denominator = pos_exp.clone()

    for neg in neg_sim:
        denominator = denominator + torch.exp(neg / tau)

    denominator = denominator + 1e-8
    return -torch.log(pos_exp / denominator + 1e-8).mean()


def compute_l_sem(sem_original, sem_para, sem_conf, sem_irr, tau: float = 0.07) -> torch.Tensor:
    sim_pos_para = cosine_similarity_matrix(sem_para, sem_original)
    sim_pos_conf = cosine_similarity_matrix(sem_conf, sem_original)
    sim_neg_irr = cosine_similarity_matrix(sem_irr, sem_original)
    pos_sim = (sim_pos_para + sim_pos_conf) / 2.0
    return infoNCE_loss(pos_sim, [sim_neg_irr], tau=tau)


def compute_l_fact(fact_original, fact_para, fact_conf, fact_irr, tau: float = 0.07) -> torch.Tensor:
    sim_pos = cosine_similarity_matrix(fact_para, fact_original)
    sim_neg_conf = cosine_similarity_matrix(fact_conf, fact_original)
    sim_neg_irr = cosine_similarity_matrix(fact_irr, fact_original)
    return infoNCE_loss(sim_pos, [sim_neg_conf, sim_neg_irr], tau=tau)


def compute_contrastive_loss(sem_original, sem_para, sem_conf, sem_irr, fact_original, fact_para, fact_conf, fact_irr, tau: float = 0.07):
    l_sem = compute_l_sem(sem_original, sem_para, sem_conf, sem_irr, tau)
    l_fact = compute_l_fact(fact_original, fact_para, fact_conf, fact_irr, tau)
    l_ctr = l_sem + l_fact
    return l_sem, l_fact, l_ctr


def validate_batch(batch_embs: Dict[str, torch.Tensor], encoder: nn.Module, projector: DualEncoderProjector, device: torch.device):
    d_sfr = encoder.config.hidden_size
    actual_bs = next(iter(batch_embs.values())).shape[0]

    for name, emb in batch_embs.items():
        assert emb.shape[1] == d_sfr, f"{name} dim mismatch"
        assert emb.shape[0] == actual_bs, f"batch size mismatch"
        assert not emb.isnan().any(), f"{name} has NaN"
        assert not emb.isinf().any(), f"{name} has Inf"

    sem_emb, fact_emb = projector(batch_embs["originals"])
    assert sem_emb.shape == (actual_bs, projector.sub_dim)
    assert fact_emb.shape == (actual_bs, projector.sub_dim)

    sem_sim = cosine_similarity_matrix(sem_emb[:2], sem_emb[:2])
    assert sem_sim.shape == (2,)
    assert sem_sim.min() >= -1.0 and sem_sim.max() <= 1.0

    logger.info("  Pre-training checks PASSED")


def train(data_file: str, encoder_name: str = SENTENCE_ENCODER_NAME, tau: float = TAU, batch_size: int = BATCH_SIZE, lr: float = LR, num_epochs: int = NUM_EPOCHS, device: torch.device = None, smoke: bool = False):
    if device is None:
        device = torch.device(DEVICE)

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    dataset = WikidataConflictDataset(data_file)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty!")

    if smoke:
        num_epochs = 2
        batch_size = min(batch_size, len(dataset))
        logger.info(f"[SMOKE] {num_epochs} epochs, batch_size={batch_size}")

    encoder, tokenizer, embed_dim = load_sentence_encoder(encoder_name, device)
    projector = DualEncoderProjector(embed_dim).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)

    logger.info(f"Trainable parameters: {sum(p.numel() for p in projector.parameters()) / 1e3:.1f}K")
    logger.info(f"Sub-dimension: {projector.sub_dim}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_conflict, drop_last=True)

    logger.info("=== Pre-training validation: 1 batch ===")
    smoke_batch = next(iter(dataloader))
    emb_dict = {
        "originals": encode_batch(encoder, tokenizer, smoke_batch["originals"], device),
        "paraphrases": encode_batch(encoder, tokenizer, smoke_batch["paraphrases"], device),
        "contradictions": encode_batch(encoder, tokenizer, smoke_batch["contradictions"], device),
        "unrelateds": encode_batch(encoder, tokenizer, smoke_batch["unrelateds"], device),
    }
    validate_batch(emb_dict, encoder, projector, device)
    logger.info(f"  SFR output shape: {emb_dict['originals'].shape}")
    logger.info(f"  Projector output shape: {projector(emb_dict['originals'][:1])[0].shape}")
    del emb_dict, smoke_batch

    global_step = 0
    for epoch in range(num_epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_conflict, drop_last=True)

        projector.train()
        epoch_loss, epoch_l_sem, epoch_l_fact = 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            emb_orig = encode_batch(encoder, tokenizer, batch["originals"], device)
            emb_para = encode_batch(encoder, tokenizer, batch["paraphrases"], device)
            emb_conf = encode_batch(encoder, tokenizer, batch["contradictions"], device)
            emb_irr = encode_batch(encoder, tokenizer, batch["unrelateds"], device)

            sem_orig, fact_orig = projector(emb_orig)
            sem_para, fact_para = projector(emb_para)
            sem_conf, fact_conf = projector(emb_conf)
            sem_irr, fact_irr = projector(emb_irr)

            l_sem, l_fact, l_ctr = compute_contrastive_loss(
                sem_orig, sem_para, sem_conf, sem_irr,
                fact_orig, fact_para, fact_conf, fact_irr, tau=tau,
            )

            optimizer.zero_grad()
            l_ctr.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += l_ctr.item()
            epoch_l_sem += l_sem.item()
            epoch_l_fact += l_fact.item()
            n_batches += 1
            global_step += 1

            with torch.no_grad():
                sim_para_orig = cosine_similarity_matrix(sem_para[:3], sem_orig[:3])
                sim_conf_orig = cosine_similarity_matrix(sem_conf[:3], sem_orig[:3])
                sim_irr_orig = cosine_similarity_matrix(sem_irr[:3], sem_orig[:3])

            pbar.set_postfix({
                "loss": f"{l_ctr.item():.4f}",
                "l_sem": f"{l_sem.item():.4f}",
                "l_fact": f"{l_fact.item():.4f}",
                "s_p": f"{sim_para_orig.mean().item():.3f}",
                "s_c": f"{sim_conf_orig.mean().item():.3f}",
                "s_i": f"{sim_irr_orig.mean().item():.3f}",
            })

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_l_sem = epoch_l_sem / max(n_batches, 1)
        avg_l_fact = epoch_l_fact / max(n_batches, 1)

        logger.info(f"Epoch {epoch + 1}/{num_epochs} | L_ctr={avg_loss:.4f} | L_sem={avg_l_sem:.4f} | L_fact={avg_l_fact:.4f}")

        if math.isnan(avg_loss):
            logger.error("Loss is NaN! Stopping training.")
            break

    save_path = CKPT_DIR / f"dual_encoder_v2__{encoder_name.replace('/', '-')}.pt"
    state = {
        "encoder_sem_state_dict": projector.encoder_sem.state_dict(),
        "encoder_fact_state_dict": projector.encoder_fact.state_dict(),
        "sub_dim": projector.sub_dim,
        "sentence_encoder_name": encoder_name,
        "d_sfr": embed_dim,
        "epoch": num_epochs,
        "train_loss": avg_loss,
        "tau": tau,
    }
    torch.save(state, save_path)
    logger.info(f"Checkpoint saved: {save_path}")
    return projector


def main():
    parser = argparse.ArgumentParser(description="Train dual encoder v2 with contrastive loss")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (2 epochs)")
    parser.add_argument("--data_file", type=str, default=str(DATA_FILE))
    parser.add_argument("--encoder_name", type=str, default=SENTENCE_ENCODER_NAME)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--tau", type=float, default=TAU, help="Temperature for contrastive loss")
    parser.add_argument("--sub_dim", type=int, default=None, help="Subspace dimension")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stage 1: Dual Encoder Training (v2)")
    logger.info(f"  Data: {args.data_file}")
    logger.info(f"  Encoder: {args.encoder_name}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  LR: {args.lr}, Tau: {args.tau}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 60)

    train(
        data_file=args.data_file,
        encoder_name=args.encoder_name,
        tau=args.tau,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
