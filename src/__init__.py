"""Speculative Decoding for Whisper

This package implements speculative decoding on OpenAI's Whisper model
to accelerate inference while maintaining output quality.
"""

from .api import SpeculativeWhisper
from .speculative_decoder import SpeculativeWhisperDecoder
from .draft_model import DraftModel, DistilWhisperDraft, LayerDropoutDraft
from .config import SpeculativeConfig

__version__ = "0.1.0"

__all__ = [
    "SpeculativeWhisper",
    "SpeculativeWhisperDecoder",
    "DraftModel",
    "DistilWhisperDraft",
    "LayerDropoutDraft",
    "SpeculativeConfig",
]
