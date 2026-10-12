"""DeclaredComputation instances: the MLP smoke instance and the decoder instance (Llama, Qwen2)."""

from .decoder import DecoderComputation
from .mlp import MLPComputation

__all__ = ["DecoderComputation", "MLPComputation"]
