"""
TCR Mixin Classes — Capability separation via multiple inheritance.

Implements the AAAI 2026 TCR paper architecture:
  - DualEncoder: semantic & factual conflict signals
  - AnswerabilityMLP: self-answerability estimation from middle layers
  - ConflictSignalProjector: [σ_sem, σ_fact, σ_ans] → embedding
  - SoftTokenEmbeddings: learnable soft tokens per signal
  - SignalEncoder: combines all components

Reference: LLaVA source code mixin pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass


# =============================================================================
# Conflict Signal Dataclass
# =============================================================================

@dataclass
class ConflictSignal:
    """
    Encapsulates conflict signals between retrieved context and model knowledge.
    Fields:
        sem_sim: semantic similarity [0, 1]
        fact_sim: factual consistency score [0, 1]
        answerability: model's self-answerability estimation [0, 1]
        extra: extensible dict for method-specific signals
    """
    sem_sim: float = 0.0
    fact_sim: float = 0.0
    answerability: float = 0.0
    extra: dict = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def to_dict(self) -> dict:
        return {
            "sem_sim": self.sem_sim,
            "fact_sim": self.fact_sim,
            "answerability": self.answerability,
            **self.extra,
        }


# =============================================================================
# Dual Encoder — two lightweight linear layers
# =============================================================================

class DualEncoder(nn.Module):
    """
    Lightweight linear projector for conflict detection.
    Two independent instances: encoder_sem and encoder_fact.

    Each takes [emb_parametric; emb_retrieved] → scalar score.
    arch: Linear(hidden_dim, hidden_dim)
    loss: L_sem for semantic, L_fact for factual
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, emb_q: torch.Tensor, emb_c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_q: parametric knowledge embedding (LLM intermediate layer rep)
            emb_c: retrieved context embedding (retriever output)
        Returns:
            scalar score (cosine similarity in projected space)
        """
        z_q = F.normalize(self.proj(emb_q), p=2, dim=-1)
        z_c = F.normalize(emb_c, p=2, dim=-1)
        return (z_q * z_c).sum(dim=-1)


# =============================================================================
# Answerability MLP — from LLM middle-layer representation
# =============================================================================

class AnswerabilityMLP(nn.Module):
    """
    Estimates whether the LLM can answer the query without retrieval.
    Uses LLM's middle-layer hidden states as features.

    arch: Linear(hidden_dim, 256) → ReLU → Linear(256, 2) → softmax → σ_ans
    loss: CrossEntropy([0, 1] labels)
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 2),
        )

    def forward(self, hidden_rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_rep: (batch, hidden_dim) — mean-pooled middle layer
        Returns:
            logits: (batch, 2)
            sigma_ans: (batch,) — probability of being answerable
        """
        logits = self.mlp(hidden_rep)
        sigma_ans = F.softmax(logits, dim=-1)[:, 1]  # P(answerable)
        return logits, sigma_ans


# =============================================================================
# Conflict Signal Projector — scalar → embedding
# =============================================================================

class ConflictSignalProjector(nn.Module):
    """
    Projects three scalar signals into LLM embedding dimension.
    This is the 'MLP Projector' in the paper.

    arch: Linear(3, 64) → ReLU → Linear(64, embed_dim)
    Input: [σ_sem, σ_fact, σ_ans] per sample
    Output: (batch, embed_dim) embedding
    """
    def __init__(self, embed_dim: int, intermediate_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, embed_dim),
        )

    def forward(
        self,
        sigma_sem: torch.Tensor,
        sigma_fact: torch.Tensor,
        sigma_ans: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sigma_sem: (batch,) or (batch, 1)
            sigma_fact: (batch,) or (batch, 1)
            sigma_ans: (batch,) or (batch, 1)
        Returns:
            (batch, embed_dim)
        """
        # Ensure 2D
        sigma_sem = sigma_sem.reshape(-1, 1)
        sigma_fact = sigma_fact.reshape(-1, 1)
        sigma_ans = sigma_ans.reshape(-1, 1)
        signals = torch.cat([sigma_sem, sigma_fact, sigma_ans], dim=-1)  # (batch, 3)
        return self.proj(signals)


# =============================================================================
# Soft Token Embeddings — learnable prefix tokens
# =============================================================================

class SoftTokenEmbeddings(nn.Module):
    """
    Learnable soft tokens per signal.

    TCR paper: each signal gets 5 soft tokens, plus 1 signal embedding = 6 tokens/signal.
    Total prefix = 5×3 + 3 = 18 tokens.

    Initialization: sample from model embedding distribution (most compatible).
    """
    def __init__(
        self,
        num_tokens_per_signal: int = 5,
        embed_dim: int = 4096,
        num_signals: int = 3,
        model_embed_weight: Optional[torch.Tensor] = None,
        init_strategy: str = "special_token",
    ):
        super().__init__()
        self.num_tokens_per_signal = num_tokens_per_signal
        self.num_signals = num_signals
        self.embed_dim = embed_dim
        total_tokens = num_tokens_per_signal * num_signals  # 15 by default

        if init_strategy == "special_token" and model_embed_weight is not None:
            # Use model's special token embeddings as initialization
            special_ids = list(range(min(5, model_embed_weight.shape[0])))
            init_weight = model_embed_weight[special_ids].mean(dim=0, keepdim=True)
            init_weight = init_weight.repeat(total_tokens, 1)
        elif init_strategy == "mean_vocab" and model_embed_weight is not None:
            init_weight = model_embed_weight.mean(dim=0, keepdim=True)
            init_weight = init_weight.repeat(total_tokens, 1)
        else:
            init_weight = torch.randn(total_tokens, embed_dim) * 0.02

        self.soft_tokens = nn.Parameter(init_weight.clone())

    def get_signal_prefix(self, signal_idx: int) -> torch.Tensor:
        """
        Returns the soft tokens for signal_idx.
        signal_idx: 0=σ_sem, 1=σ_fact, 2=σ_ans
        Returns: (num_tokens_per_signal, embed_dim)
        """
        start = signal_idx * self.num_tokens_per_signal
        end = start + self.num_tokens_per_signal
        return self.soft_tokens[start:end]

    @property
    def prefix_length(self) -> int:
        """Total length of soft token prefix = num_tokens × num_signals"""
        return self.num_tokens_per_signal * self.num_signals


# =============================================================================
# Signal Encoder — combines all TCR components
# =============================================================================

class SignalEncoder(nn.Module):
    """
    Combines all TCR signal processing components:
    - DualEncoder (sem + fact)
    - AnswerabilityMLP
    - ConflictSignalProjector
    - SoftTokenEmbeddings

    Also handles embedding extraction from LLM middle layers.
    """
    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        num_soft_tokens: int = 5,
        init_from_model: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Dual encoders — independent, trained separately
        self.encoder_sem = DualEncoder(hidden_dim)
        self.encoder_fact = DualEncoder(hidden_dim)

        # Answerability head
        self.answerability_mlp = AnswerabilityMLP(hidden_dim)

        # Signal projector: [σ_sem, σ_fact, σ_ans] → embedding
        self.signal_projector = ConflictSignalProjector(embed_dim)

        # Soft tokens
        embed_weight = init_from_model.weight.data if init_from_model is not None else None
        self.soft_tokens = SoftTokenEmbeddings(
            num_tokens_per_signal=num_soft_tokens,
            embed_dim=embed_dim,
            num_signals=3,
            model_embed_weight=embed_weight,
            init_strategy="special_token" if embed_weight is not None else "randn",
        )

    def extract_middle_layer_rep(
        self,
        llm,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Extract middle-layer representation from LLM for answerability estimation.

        For decoder-only models: mean-pool the middle layer over sequence.
        """
        with torch.no_grad():
            outputs = llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            layer_idx = int(len(hidden_states) * layer_ratio)
            hidden = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            rep = hidden.mean(dim=1)  # (batch, hidden_dim)
        return rep

    def forward_signals(
        self,
        emb_parametric: torch.Tensor,
        emb_retrieved: torch.Tensor,
        hidden_rep: torch.Tensor,
    ) -> Tuple[ConflictSignal, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all three signals.

        Args:
            emb_parametric: (batch, hidden_dim) — LLM middle-layer rep
            emb_retrieved: (batch, hidden_dim) — retriever embedding
            hidden_rep: (batch, hidden_dim) — same as emb_parametric for convenience

        Returns:
            signal: ConflictSignal dataclass
            sigma_sem: (batch,)
            sigma_fact: (batch,)
            sigma_ans: (batch,)
        """
        sigma_sem = self.encoder_sem(emb_parametric, emb_retrieved)
        sigma_fact = self.encoder_fact(emb_parametric, emb_retrieved)

        _, sigma_ans = self.answerability_mlp(hidden_rep)

        signal = ConflictSignal(
            sem_sim=sigma_sem.mean().item() if sigma_sem.numel() == 1 else sigma_sem.mean().item(),
            fact_sim=sigma_fact.mean().item() if sigma_fact.numel() == 1 else sigma_fact.mean().item(),
            answerability=sigma_ans.mean().item() if sigma_ans.numel() == 1 else sigma_ans.mean().item(),
            extra={
                "sigma_sem": sigma_sem.detach(),
                "sigma_fact": sigma_fact.detach(),
                "sigma_ans": sigma_ans.detach(),
            }
        )
        return signal, sigma_sem, sigma_fact, sigma_ans


# =============================================================================
# Attention Mask Utilities
# =============================================================================

def build_augmented_attention_mask(
    original_attention_mask: torch.Tensor,
    prefix_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build attention mask for augmented inputs (with soft-token prefix).

    The prefix is always 'real' content — all 1s.
    Original padding positions stay 0.

    Args:
        original_attention_mask: (batch, seq_len) — 1=valid, 0=padding
        prefix_len: number of soft-token positions added
        device

    Returns:
        (batch, prefix_len + seq_len)
    """
    batch_size = original_attention_mask.shape[0]
    prefix_mask = torch.ones(batch_size, prefix_len, dtype=torch.long, device=device)
    return torch.cat([prefix_mask, original_attention_mask], dim=1)


# =============================================================================
# Abstract Mixin Interfaces
# =============================================================================

class ConflictDetectorMixin(ABC):
    """Computes semantic and factual conflict signals."""

    @abstractmethod
    def detect_conflict(
        self,
        query: Union[str, List[str]],
        context: Union[str, List[str]],
    ) -> ConflictSignal:
        raise NotImplementedError

    @abstractmethod
    def compute_signals(
        self,
        emb_parametric: torch.Tensor,
        emb_retrieved: torch.Tensor,
        hidden_rep: torch.Tensor,
    ) -> Tuple[ConflictSignal, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SelfAnswerabilityMixin(ABC):
    """Estimates whether the model can answer the query without retrieval."""

    @abstractmethod
    def estimate_answerability(
        self,
        query: Union[str, List[str]],
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def ppl_to_answerability(ppl: float) -> float:
        """Convert perplexity to answerability score (0~1)."""
        return float(1.0 / (1.0 + ppl / 10.0))


class SoftPromptMixin(ABC):
    """Injects conflict signals into soft-token-prefixed embeddings."""

    @abstractmethod
    def build_augmented_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        signal: ConflictSignal,
        sigma_sem: torch.Tensor,
        sigma_fact: torch.Tensor,
        sigma_ans: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build augmented embeddings with soft-token prefix.

        Returns:
            (augmented_inputs_embeds, augmented_attention_mask)
        """
        raise NotImplementedError
