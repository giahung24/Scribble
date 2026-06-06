"""Pluggable STT providers. v1 exposes a local OpenAI-compatible provider for
the batch Upload pipeline; the interface is shaped to extend to realtime in v2."""
from .base import STTSegment, STTProvider, segments_to_text
from .local_openai import LocalOpenAIProvider

__all__ = ["STTSegment", "STTProvider", "segments_to_text", "LocalOpenAIProvider"]
