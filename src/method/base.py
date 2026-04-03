"""
Base classes for TCR project.
Capability separation via Mixin pattern (LLaVA-style).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch

# Import ConflictSignal from mixins (shared dataclass)
from src.method.mixins import ConflictSignal


class BaseMethod(ABC):
    """
    Minimal interface for TCR methods.
    All concrete implementations inherit this.
    """

    def __init__(self, llm: Any, encoder: Any, cfg: dict):
        self.llm = llm
        self.encoder = encoder
        self.cfg = cfg
        self._setup()

    def _setup(self):
        """Hook for subclass-specific initialization."""
        pass

    @abstractmethod
    def detect(
        self,
        query: Union[str, List[str]],
        context: Union[str, List[str]],
    ) -> ConflictSignal:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        query: Union[str, List[str]],
        context: Union[str, List[str]],
        signal: Optional[ConflictSignal] = None,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return all trainable parameters across LLM and encoder."""
        params = []
        for model in [self.llm, self.encoder]:
            if model is not None:
                params.extend(p for p in model.parameters() if p.requires_grad)
        return params

    def state_dict(self) -> dict:
        state = {}
        if self.llm is not None:
            state["llm"] = self.llm.state_dict()
        if self.encoder is not None:
            state["encoder"] = self.encoder.state_dict()
        return state

    def load_state_dict(self, state_dict: dict):
        if "llm" in state_dict and self.llm is not None:
            self.llm.load_state_dict(state_dict["llm"])
        if "encoder" in state_dict and self.encoder is not None:
            self.encoder.load_state_dict(state_dict["encoder"])


class BaseEncoder(ABC):
    """Base for retrieval encoder (SFR, ColBERT, E5, etc.)."""

    def __init__(self, model_name_or_path: str, cfg: dict):
        self.model_name_or_path = model_name_or_path
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._hidden_size = None
        self._embed_length = None

    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, query: str, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_doc(self, doc: str, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_embed_dim(self) -> int:
        if self._hidden_size is None:
            self._infer_dims()
        return self._hidden_size

    def get_embed_length(self) -> int:
        if self._embed_length is None:
            self._infer_dims()
        return self._embed_length

    def _infer_dims(self):
        self._hidden_size = 128
        self._embed_length = 1

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer
