"""
TCR Method — Transparent Knowledge Conflict Handling in RAG.

Implements the AAAI 2026 paper architecture:
  1. DualEncoder → σ_sem, σ_fact (semantic & factual conflict signals)
  2. AnswerabilityMLP → σ_ans (self-answerability from LLM middle layers)
  3. ConflictSignalProjector → signal embedding (3 scalars → LLM embed dim)
  4. SoftTokenEmbeddings → learnable prefix tokens (5 per signal)
  5. Augmented embedding construction with prefix concatenation

LLM backbone is ALWAYS frozen during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from src.method.base import BaseMethod
from src.method.mixins import (
    ConflictSignal,
    ConflictDetectorMixin,
    SelfAnswerabilityMixin,
    SoftPromptMixin,
    SignalEncoder,
    build_augmented_attention_mask,
)
from src.registry import METHOD_REGISTRY


__all__ = ["TCRMethod"]


@METHOD_REGISTRY.register("tcr")
class TCRMethod(ConflictDetectorMixin, SelfAnswerabilityMixin, SoftPromptMixin, BaseMethod):
    """
    TCR: Transparent Conflict Resolution for RAG.

    Combines ConflictDetectorMixin + SelfAnswerabilityMixin + SoftPromptMixin.

    Training path:
        detect(query, context) → ConflictSignal
            → compute_signals(emb_parametric, emb_retrieved, hidden_rep)
                → σ_sem, σ_fact, σ_ans
        build_augmented_embeddings(...) → augmented inputs_embeds
        llm.forward(inputs_embeds=augmented_embeds)
        compute_loss(logits, labels, prefix_len)

    Inference path:
        detect() → ConflictSignal
        decision: use_self_answer vs use_retrieval
        build_augmented_embeddings() → llm.generate(inputs_embeds=...)
    """

    def __init__(self, llm: Any, encoder: Any, cfg: dict):
        self._cfg = cfg
        self.device = cfg.get("device", "cpu")
        self._tokenizer = None

        # TCR-specific config
        self.num_soft_tokens = cfg.get("num_soft_tokens", 5)
        self.middle_layer_ratio = cfg.get("middle_layer_ratio", 0.5)
        self.conflict_threshold = cfg.get("conflict_threshold", 0.3)
        self.use_self_answerability = cfg.get("use_self_answerability", True)

        # Signal encoder (all TCR components)
        self._signal_encoder: Optional[SignalEncoder] = None
        self._signal_encoder_loaded = False

        super().__init__(llm, encoder, cfg)

    def _setup(self):
        """Post-init: freeze LLM backbone."""
        if self.llm is not None:
            for param in self.llm.parameters():
                param.requires_grad = False

    # -------------------------------------------------------------------------
    # Signal Encoder (lazy initialization)
    # -------------------------------------------------------------------------
    @property
    def signal_encoder(self) -> Optional[SignalEncoder]:
        if self._signal_encoder_loaded:
            return self._signal_encoder
        if self.llm is None:
            return None

        hidden_dim = getattr(self.llm.config, "hidden_size", 4096)
        embed_dim = hidden_dim

        # Get model embedding layer for soft token initialization
        embed_layer = None
        try:
            embed_layer = self.llm.get_input_embeddings()
        except Exception:
            pass

        self._signal_encoder = SignalEncoder(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_soft_tokens=self.num_soft_tokens,
            init_from_model=embed_layer,
        ).to(self.device)
        self._signal_encoder_loaded = True
        return self._signal_encoder

    def _ensure_signal_encoder(self):
        _ = self.signal_encoder

    # -------------------------------------------------------------------------
    # Encoder helpers
    # -------------------------------------------------------------------------
    def _encode_query(self, queries: List[str]) -> torch.Tensor:
        if self.encoder is None:
            return torch.zeros(len(queries), 128, device=self.device)
        inputs = self.encoder.tokenizer(
            queries, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            return self.encoder.encode_query(ids, mask)

    def _encode_doc(self, docs: List[str]) -> torch.Tensor:
        if self.encoder is None:
            return torch.zeros(len(docs), 128, device=self.device)
        inputs = self.encoder.tokenizer(
            docs, padding=True, truncation=True, max_length=180, return_tensors="pt"
        )
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            return self.encoder.encode_doc(ids, mask)

    def _encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts for retrieval embedding (no gradient)."""
        if self.encoder is None:
            return torch.zeros(len(texts), 128, device=self.device)
        inputs = self.encoder.tokenizer(
            texts, padding=True, truncation=True, max_length=180, return_tensors="pt"
        )
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            return self.encoder.encode(ids, mask)

    # -------------------------------------------------------------------------
    # ConflictDetectorMixin
    # -------------------------------------------------------------------------
    def detect(
        self,
        query: Union[str, List[str]],
        context: Union[str, List[str]],
    ) -> ConflictSignal:
        """Detect conflict signals between query and retrieved context."""
        if isinstance(query, str):
            query = [query]
        if isinstance(context, str):
            context = [context]

        query_embeds = self._encode_query(query)
        context_embeds = self._encode_doc(context)

        # Compute σ_sem, σ_fact via dual encoders
        se = self.signal_encoder
        sigma_sem = se.encoder_sem(query_embeds, context_embeds)
        sigma_fact = se.encoder_fact(query_embeds, context_embeds)

        # Compute σ_ans via LLM middle layer (only if self-answerability enabled)
        answerability = 0.0
        sigma_ans_scalar = torch.zeros(len(query), device=self.device)
        if self.use_self_answerability:
            sigma_ans_scalar = self._estimate_answerability_tensor(query)

        # Build ConflictSignal
        conflict_score = 1.0 - (sigma_sem.mean().item() * 0.5 + sigma_fact.mean().item() * 0.5)

        return ConflictSignal(
            sem_sim=sigma_sem.mean().item(),
            fact_sim=sigma_fact.mean().item(),
            answerability=sigma_ans_scalar.mean().item(),
            extra={
                "conflict_score": conflict_score,
                "sigma_sem": sigma_sem.detach(),
                "sigma_fact": sigma_fact.detach(),
                "sigma_ans": sigma_ans_scalar.detach(),
                "use_retrieval": conflict_score > self.conflict_threshold,
                "use_self_answer": sigma_ans_scalar.mean().item() > conflict_score,
            }
        )

    def compute_signals(
        self,
        emb_parametric: torch.Tensor,
        emb_retrieved: torch.Tensor,
        hidden_rep: torch.Tensor,
    ) -> Tuple[ConflictSignal, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all three signals from embeddings."""
        se = self.signal_encoder
        return se.forward_signals(emb_parametric, emb_retrieved, hidden_rep)

    # -------------------------------------------------------------------------
    # SelfAnswerabilityMixin
    # -------------------------------------------------------------------------
    def estimate_answerability(self, query: Union[str, List[str]]) -> float:
        """Estimate answerability for a single query string."""
        vals = self._estimate_answerability_tensor(query if isinstance(query, list) else [query])
        return vals.mean().item() if vals.numel() > 0 else 0.5

    def _estimate_answerability_tensor(self, queries: List[str]) -> torch.Tensor:
        """Estimate answerability as a tensor for a batch of queries."""
        tokenizer = self._get_tokenizer()
        if tokenizer is None or self.llm is None:
            return torch.full((len(queries),), 0.5, device=self.device)

        inputs = tokenizer(
            queries, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            se = self.signal_encoder
            hidden_rep = se.extract_middle_layer_rep(
                self.llm, ids, mask, layer_ratio=self.middle_layer_ratio
            )
            _, sigma_ans = se.answerability_mlp(hidden_rep)

        return sigma_ans

    # -------------------------------------------------------------------------
    # SoftPromptMixin
    # -------------------------------------------------------------------------
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

        Structure: [soft_sem×5 | sig_sem | soft_fact×5 | sig_fact | soft_ans×5 | sig_ans | original_embeds]
        Total prefix = 5*3 + 3 = 18 tokens

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            signal: ConflictSignal
            sigma_sem: (batch,) or (batch, 1)
            sigma_fact: (batch,) or (batch, 1)
            sigma_ans: (batch,) or (batch, 1)

        Returns:
            augmented_inputs_embeds: (batch, prefix_len + seq_len, embed_dim)
            augmented_attention_mask: (batch, prefix_len + seq_len)
        """
        se = self.signal_encoder
        B = input_ids.shape[0]
        device = input_ids.device

        # Original input embeddings
        embed_layer = self.llm.get_input_embeddings()
        x = embed_layer(input_ids)  # (B, seq_len, embed_dim)

        # Project signals to embeddings
        sig_sem_emb = se.signal_projector(sigma_sem, torch.zeros_like(sigma_sem), torch.zeros_like(sigma_sem))
        sig_fact_emb = se.signal_projector(torch.zeros_like(sigma_fact), sigma_fact, torch.zeros_like(sigma_fact))
        sig_ans_emb = se.signal_projector(torch.zeros_like(sigma_ans), torch.zeros_like(sigma_ans), sigma_ans)

        # Soft token prefixes
        soft_sem = se.soft_tokens.get_signal_prefix(0).unsqueeze(0).expand(B, -1, -1)  # (B, 5, D)
        soft_fact = se.soft_tokens.get_signal_prefix(1).unsqueeze(0).expand(B, -1, -1)
        soft_ans = se.soft_tokens.get_signal_prefix(2).unsqueeze(0).expand(B, -1, -1)

        # Signal embeddings: (B, 1, D)
        sig_sem_emb = sig_sem_emb.unsqueeze(1)
        sig_fact_emb = sig_fact_emb.unsqueeze(1)
        sig_ans_emb = sig_ans_emb.unsqueeze(1)

        # Concatenate: [soft×5 | signal_emb] × 3 + original
        prefix_len = se.soft_tokens.prefix_length + 3  # 15 + 3 = 18
        x_aug = torch.cat([
            soft_sem,  sig_sem_emb,   # (B, 6, D)
            soft_fact, sig_fact_emb,  # (B, 6, D)
            soft_ans,  sig_ans_emb,   # (B, 6, D)
            x,                        # (B, seq_len, D)
        ], dim=1)  # → (B, 18+seq_len, D)

        # Build attention mask (prefix all 1s)
        aug_mask = build_augmented_attention_mask(attention_mask, prefix_len, device)

        return x_aug, aug_mask

    def inject_soft_prompt(
        self,
        signal: ConflictSignal,
        input_embeds: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy compatibility — redirects to build_augmented_embeddings."""
        raise NotImplementedError("Use build_augmented_embeddings instead")

    # -------------------------------------------------------------------------
    # Generation (Inference)
    # -------------------------------------------------------------------------
    def generate(
        self,
        query: Union[str, List[str]],
        context: Union[str, List[str]],
        signal: Optional[ConflictSignal] = None,
        **kwargs,
    ) -> Any:
        mode = kwargs.get("mode", "inference")
        if mode == "train":
            return self._forward_train(query, context, signal, **kwargs)
        else:
            return self._forward_inference(query, context, signal, **kwargs)

    def _forward_train(
        self,
        query: Union[str, List[str]],
        context: Union[str, List[str]],
        signal: Optional[ConflictSignal],
        **kwargs,
    ) -> Dict[str, Any]:
        """Training forward pass with TCR augmented embeddings."""
        self._ensure_signal_encoder()

        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        labels = kwargs.get("labels")

        if input_ids is None:
            return {"loss": None}

        # Encode context to retrieval embeddings
        retrieval_embeds = self._encode(context)  # (B, retriever_hidden)
        retrieval_embeds = retrieval_embeds.to(self.device)

        # Get LLM middle-layer representation
        se = self.signal_encoder
        hidden_rep = se.extract_middle_layer_rep(
            self.llm, input_ids, attention_mask, layer_ratio=self.middle_layer_ratio
        )

        # Compute signals
        sig, sigma_sem, sigma_fact, sigma_ans = self.compute_signals(
            hidden_rep, retrieval_embeds, hidden_rep
        )

        # Build augmented embeddings with soft-token prefix
        aug_embeds, aug_mask = self.build_augmented_embeddings(
            input_ids, attention_mask, sig, sigma_sem, sigma_fact, sigma_ans
        )

        # Forward through LLM
        outputs = self.llm(inputs_embeds=aug_embeds, attention_mask=aug_mask)
        logits = outputs.logits

        # Compute loss with proper masking
        loss = self.compute_loss(logits, labels, aug_mask, prefix_len=se.soft_tokens.prefix_length + 3)

        return {"loss": loss, "logits": logits}

    def _forward_inference(
        self,
        query: Union[str, List[str]],
        context: Union[str, List[str]],
        signal: Optional[ConflictSignal],
        **kwargs,
    ) -> Any:
        """Inference forward pass with TCR augmented embeddings."""
        self._ensure_signal_encoder()

        max_new_tokens = kwargs.get("max_new_tokens", 100)
        do_sample = kwargs.get("do_sample", False)
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")

        # If no signal provided, compute it
        if signal is None:
            signal = self.detect(query, context)

        use_self = signal.extra.get("use_self_answer", False)

        # If model can answer without retrieval
        if use_self:
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

        # Retrieval path
        retrieval_embeds = self._encode(context)
        retrieval_embeds = retrieval_embeds.to(self.device)

        # Get LLM middle-layer representation
        se = self.signal_encoder
        hidden_rep = se.extract_middle_layer_rep(
            self.llm, input_ids, attention_mask, layer_ratio=self.middle_layer_ratio
        )

        # Compute signals
        sig, sigma_sem, sigma_fact, sigma_ans = self.compute_signals(
            hidden_rep, retrieval_embeds, hidden_rep
        )

        # Build augmented embeddings
        aug_embeds, aug_mask = self.build_augmented_embeddings(
            input_ids, attention_mask, sig, sigma_sem, sigma_fact, sigma_ans
        )

        return self.llm.generate(
            inputs_embeds=aug_embeds,
            attention_mask=aug_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    # -------------------------------------------------------------------------
    # Loss Computation (with proper prefix masking)
    # -------------------------------------------------------------------------
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_len: int = 18,
    ) -> torch.Tensor:
        """
        Compute NLL loss with proper masking:
          1. Mask prefix positions (soft tokens)
          2. Mask padding tokens
          3. Mask query / prompt part (where labels == -100)

        Args:
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len) — already shifted, -100 for no-loss
            attention_mask: (batch, seq_len)
            prefix_len: number of prefix positions (default 18 = 5*3 + 3)
        """
        if labels is None:
            return torch.tensor(0.0, device=logits.device)

        B, Seq, V = logits.shape
        device = logits.device

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()  # (B, Seq-1, V)
        shift_labels = labels[..., 1:].contiguous()        # (B, Seq-1)

        # Build loss mask
        loss_mask = torch.ones_like(shift_labels, dtype=torch.bool, device=device)

        # 1. Mask prefix positions
        loss_mask[:, :prefix_len] = False

        # 2. Mask padding (where attention_mask says 0)
        # attention_mask is (B, Seq), shift it too
        shift_mask = attention_mask[..., 1:].contiguous()
        loss_mask = loss_mask & (shift_mask.bool())

        # 3. Mask where labels already say -100 (query part)
        loss_mask = loss_mask & (shift_labels != -100)

        # Compute cross-entropy only on masked positions
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss_per_token = loss_fct(
            shift_logits.view(-1, V),
            shift_labels.view(-1)
        )
        loss = (loss_per_token * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)

        return loss

    # -------------------------------------------------------------------------
    # SNR Dynamic Weighting (optional)
    # -------------------------------------------------------------------------
    def compute_snr_weights(
        self,
        predictions: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SNR-based dynamic weights for each signal.
        Paper: use signal-to-noise ratio to weight contributions.
        """
        snr_list = []
        for pred in predictions:
            var_pred = pred.var()
            var_noise = (labels.float() - pred.float()).var()
            snr = var_pred / (var_noise + 1e-8)
            snr_list.append(snr)
        snr_tensor = torch.stack(snr_list)
        return torch.softmax(snr_tensor, dim=0)

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if hasattr(self.llm, "tokenizer"):
            return self.llm.tokenizer
        if hasattr(self.llm, "get_input_embeddings"):
            try:
                from transformers import AutoTokenizer
                cfg = getattr(self.llm, "config", None)
                name = getattr(cfg, "name_or_path", None) if cfg else None
                self._tokenizer = AutoTokenizer.from_pretrained(
                    name or "mistralai/Mistral-7B-Instruct-v0.2", use_fast=False
                )
                return self._tokenizer
            except Exception:
                pass
        return None

    # -------------------------------------------------------------------------
    # TCR Checkpoint Save/Load
    # -------------------------------------------------------------------------
    def get_tcr_modules(self) -> Dict[str, nn.Module]:
        """
        Returns only the trainable TCR components (for checkpointing).
        Excludes the frozen LLM backbone.
        """
        if self._signal_encoder is None:
            self._ensure_signal_encoder()
        return {
            "signal_encoder": self._signal_encoder,
        }

    def save_tcr_checkpoint(self, path: str):
        """Save only trainable TCR components."""
        modules = self.get_tcr_modules()
        state = {k: v.state_dict() for k, v in modules.items()}
        torch.save(state, path)

    def load_tcr_checkpoint(self, path: str, strict: bool = True):
        """Load TCR checkpoint (does NOT touch LLM backbone)."""
        ckpt = torch.load(path, map_location="cpu")
        modules = self.get_tcr_modules()
        for k, v in modules.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k], strict=strict)
