"""
DataLoader for TCR project.
"""

import random
import copy
import torch
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List
from torch.utils.data import DataLoader, Dataset

from src.registry import DATALOADER_REGISTRY

__all__ = ["build_dataloader"]


class BaseDataLoader(ABC):
    """Base for all dataloaders."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.smoke_size = cfg.get("smoke_size", 10)
        self.data = None

    @abstractmethod
    def load(self, smoke: bool = False) -> Dataset:
        raise NotImplementedError

    def _apply_smoke(self, data: Any, smoke: bool) -> Any:
        if smoke and len(data) > self.smoke_size:
            print(f"[SMOKE] {len(data)} samples truncated to {self.smoke_size}")
            return data[:self.smoke_size]
        return data


class TCRDataset(Dataset):
    """Dataset that yields pre-encoded samples."""

    def __init__(self, data: List[dict], encode_fn: Callable):
        self.data = data
        self.encode_fn = encode_fn

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.encode_fn(self.data[idx])


class BaseCollator:
    """Base collator for batch assembly."""

    def __init__(self, llm_tokenizer, retriever_tokenizer=None, retrieval_context_length: int = 180):
        self.llm_tokenizer = llm_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.retrieval_context_length = retrieval_context_length

    def _pad(self, input_ids, labels=None, padding_side='right'):
        pad_id = self.llm_tokenizer.pad_token_id if self.llm_tokenizer.pad_token_id is not None else 0

        def __pad(ids, p_id, p_side):
            if p_side == 'left':
                flipped = [torch.flip(x, dims=[0]) for x in ids]
                padded = torch.nn.utils.rnn.pad_sequence(flipped, batch_first=True, padding_value=p_id)
                return torch.flip(padded, dims=[1])
            return torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=p_id)

        padded_ids = __pad(input_ids, pad_id, padding_side)
        attn_mask = (padded_ids != pad_id).long()
        if labels is not None:
            padded_labels = __pad(labels, -100, padding_side)
            return padded_ids, attn_mask, padded_labels
        return padded_ids, attn_mask, None


@DATALOADER_REGISTRY.register("pretrain")
class PretrainDataLoader(BaseDataLoader):
    """Paraphrase pretraining dataloader. Format: {"text": "..."}"""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.data_path = cfg.get("train_file", "data/pretrain.jsonl")
        self._load_raw_data()

    def _load_raw_data(self):
        try:
            import json
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.data = [json.loads(line) for line in f]
        except Exception as e:
            print(f"[WARNING] Failed to load {self.data_path}: {e}")
            self.data = []

    def load(self, smoke: bool = False) -> TCRDataset:
        from src.dataloader.preprocessing import encode_with_chat_format_pretrain
        return TCRDataset(self._apply_smoke(self.data, smoke), encode_with_chat_format_pretrain)


@DATALOADER_REGISTRY.register("finetune")
class FinetuneDataLoader(BaseDataLoader):
    """Instruction-tuning dataloader. Format: {"messages": [...], "background": "...", "task_type": "..."}"""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.data_path = cfg.get("train_file", "data/finetune.jsonl")
        self.use_rag_tuning = cfg.get("use_rag_tuning", True)
        self._load_raw_data()

    def _load_raw_data(self):
        try:
            import json
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.data = [json.loads(line) for line in f]
        except Exception as e:
            print(f"[WARNING] Failed to load {self.data_path}: {e}")
            self.data = []

    def load(self, smoke: bool = False) -> TCRDataset:
        from src.dataloader.preprocessing import encode_with_chat_format_finetune
        return TCRDataset(self._apply_smoke(self.data, smoke), encode_with_chat_format_finetune)


def collator(samples, llm_tokenizer, retriever_tokenizer=None, retrieval_context_length=180):
    """
    Collate function for TCR training.
    Output keys: tcr_input_ids, tcr_labels, [input_ids, labels], retriever_input_ids
    """
    pad_side = getattr(llm_tokenizer, 'padding_side', 'right') or 'right'
    pad_id = llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else 0

    def _pad(ids, label_ids=None):
        if pad_side == 'left':
            flipped = [torch.flip(x, dims=[0]) for x in ids]
            padded = torch.nn.utils.rnn.pad_sequence(flipped, batch_first=True, padding_value=pad_id)
            padded = torch.flip(padded, dims=[1])
        else:
            padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        attn = (padded != pad_id).long()
        if label_ids is not None:
            if pad_side == 'left':
                lpadded = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)
                lpadded = torch.flip(lpadded, dims=[1])
            else:
                lpadded = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)
            return padded, attn, lpadded
        return padded, attn, None

    tcr_ids, tcr_attn, tcr_labels = _pad(
        [s['tcr_input_ids'] for s in samples],
        [s['tcr_labels'] for s in samples],
    )
    ret = {
        "tcr_input_ids": tcr_ids,
        "tcr_attention_mask": tcr_attn,
        "tcr_labels": tcr_labels,
    }

    if 'retriever_input_text' in samples[0] and retriever_tokenizer is not None:
        flat_text = [x for y in [s['retriever_input_text'] for s in samples] for x in y]
        tok = retriever_tokenizer(
            flat_text,
            max_length=retrieval_context_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        ret['retriever_input_ids'] = tok['input_ids']
        ret['retriever_attention_mask'] = tok['attention_mask']
        ret['retriever_input_text'] = flat_text

    if 'input_ids' in samples[0]:
        ids, attn, labels = _pad(
            [s['input_ids'] for s in samples],
            [s['labels'] for s in samples],
        )
        ret['input_ids'] = ids
        ret['attention_mask'] = attn
        ret['labels'] = labels

    return ret


def build_dataloader(
    name: str,
    cfg: dict,
    tokenizer,
    retriever_tokenizer=None,
    smoke: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """Build dataloader by name."""
    loader_cls = DATALOADER_REGISTRY.get(name)
    loader = loader_cls(cfg)
    dataset = loader.load(smoke=smoke)
    collate_fn = partial(
        collator,
        llm_tokenizer=tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retrieval_context_length=cfg.get("retrieval_context_length", 180),
    )
    return DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=cfg.get("per_device_train_batch_size", 1),
        num_workers=cfg.get("preprocessing_num_workers", 0),
    )
