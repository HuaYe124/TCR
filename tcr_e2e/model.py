"""TCR Model components shared by train.py and eval.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalProjector(nn.Module):
    """Project 3 scalar signals to embedding dimension."""

    def __init__(self, embed_dim=4096, intermediate_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(3, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 3 * embed_dim),
        )

    def forward(self, sigma_sem, sigma_fact, sigma_ans):
        signals = torch.stack([sigma_sem, sigma_fact, sigma_ans], dim=-1)
        proj_out = self.proj(signals)
        return proj_out.view(-1, 3, self.embed_dim)


class TCRModel(nn.Module):
    """TCR Model: Frozen LLM + trainable SignalProjector."""

    def __init__(self, llm, embed_dim, num_soft_tokens=5):
        super().__init__()
        self.llm = llm
        self.embed_dim = embed_dim
        self.signal_projector = SignalProjector(embed_dim)
        self.prefix_len = 3
        self._pad_token_id = None

    def get_trainable_params(self):
        return list(self.signal_projector.parameters())

    def build_augmented_embeddings(self, input_ids, sigma_sem, sigma_fact, sigma_ans, attention_mask, device, tokenizer=None):
        B = input_ids.size(0)
        embed_layer = self.llm.get_input_embeddings()
        orig_emb = embed_layer(input_ids.to(device))

        signal_emb = self.signal_projector(sigma_sem, sigma_fact, sigma_ans)
        sig_sem = signal_emb[:, 0]
        sig_fact = signal_emb[:, 1]
        sig_ans = signal_emb[:, 2]

        if tokenizer is not None:
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0
        else:
            pad_token_id = getattr(self, '_pad_token_id', 0) or 0

        pad_emb = embed_layer.weight[pad_token_id].unsqueeze(0).unsqueeze(0).expand(B, 3, -1)

        alpha = 0.5
        prefix_emb = torch.cat([
            pad_emb[:, 0:1, :] + alpha * sig_sem.unsqueeze(1),
            pad_emb[:, 1:2, :] + alpha * sig_fact.unsqueeze(1),
            pad_emb[:, 2:3, :] + alpha * sig_ans.unsqueeze(1),
        ], dim=1)

        augmented = torch.cat([prefix_emb, orig_emb], dim=1)

        attention_mask = attention_mask.to(device)
        aug_mask = torch.cat([
            torch.ones(B, self.prefix_len, dtype=torch.long, device=device),
            attention_mask
        ], dim=1)

        return augmented, aug_mask

    def forward(self, input_ids, labels, attention_mask, sigma_sem, sigma_fact, sigma_ans, device, tokenizer=None):
        B = input_ids.size(0)
        aug_emb, aug_mask = self.build_augmented_embeddings(
            input_ids, sigma_sem, sigma_fact, sigma_ans, attention_mask, device, tokenizer
        )
        aug_emb = aug_emb.to(self.llm.dtype)

        outputs = self.llm(inputs_embeds=aug_emb, attention_mask=aug_mask)
        logits = outputs.logits

        prefix_ignore = torch.full((B, self.prefix_len), -100, dtype=labels.dtype, device=device)
        aug_labels = torch.cat([prefix_ignore, labels], dim=1)

        shift_logits = logits[:, :-1, :]
        shift_labels = aug_labels[:, 1:]

        V = logits.shape[-1]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        return {"loss": loss, "logits": logits}
