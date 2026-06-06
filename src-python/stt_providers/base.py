"""Normalized STT output + provider interface (capability + session model)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class STTSegment:
    """One transcribed span. start_ms/end_ms are None when the provider gives
    no timestamps. speaker_id is None unless the provider diarizes natively
    (the batch pipeline assigns speakers via CAM++ when it's None)."""
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    speaker_id: int | None = None
    is_final: bool = True
    language: str | None = None


def segments_to_text(segments: list[STTSegment]) -> str:
    """Flatten segments into a single transcript string for the batch
    chunk path (chunk-level timestamps come from the VAD plan, not here)."""
    return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()


class STTProvider(ABC):
    name: str = "base"
    supports_batch: bool = False
    supports_realtime: bool = False
    native_diarization: bool = False
    native_translation: bool = False

    @abstractmethod
    def transcribe_file(self, wav_path: Path | str, language: str, **opts) -> list[STTSegment]:
        """Batch transcription: a WAV file in, normalized segments out."""
        raise NotImplementedError

    def open_session(self, language: str, **opts) -> "STTSession":
        """Realtime session — declared for v2; v1 providers may not implement it."""
        raise NotImplementedError(f"{self.name} does not support realtime in v1")


class STTSession(ABC):
    """Realtime session interface. Declared for v2; not implemented in v1."""

    @abstractmethod
    def feed_audio(self, pcm_bytes: bytes) -> None: ...

    @abstractmethod
    def results(self) -> Iterator[STTSegment]: ...

    @abstractmethod
    def stop(self) -> None: ...
