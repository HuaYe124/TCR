"""
Registry system for TCR project.
Reference: LLaVA source code mixin + registry pattern.
"""

from typing import Any, Callable, Dict, Type


class Registry:
    """Generic registry: name -> class mapping."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}
        self._lazy_imports: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: str | None = None) -> Callable[[Type], Type]:
        """Decorator to register a class."""
        def decorator(cls: Type) -> Type:
            key = name if name is not None else cls.__name__.lower()
            assert key not in self._registry, (
                f"[{self._name}] '{key}' already registered. Available: {list(self._registry)}"
            )
            self._registry[key] = cls
            cls._registry_name = key
            return cls
        return decorator

    def get(self, name: str) -> Type:
        if name not in self._registry:
            raise KeyError(
                f"[{self._name}] '{name}' not registered. Available: {list(self._registry)}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs) -> Any:
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


METHOD_REGISTRY = Registry("method")
MODEL_REGISTRY = Registry("model")
RETRIEVER_REGISTRY = Registry("retriever")
DATALOADER_REGISTRY = Registry("dataloader")
