"""Back-compat wrapper. Prefer atom.hybrid_model.HybridTransformer."""

from .hybrid_model import HybridTransformer, HybridMistralFromGGUF, ModelConfig

__all__ = ["HybridTransformer", "HybridMistralFromGGUF", "ModelConfig"]
