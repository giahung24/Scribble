"""Streaming Whisper ASR: re-decode a rolling buffer and commit stable words.

feed(pcm_bytes) appends int16 PCM, re-decodes the whole buffer with Whisper,
flattens the word list, and runs it through LocalAgreement → (newly_committed,
partial). endpoint() commits whatever remains and resets for the next segment.

The real Whisper call is injected as `transcribe_fn(float32_audio) -> segments`
(a list with `.words` of objects having `.word/.start/.end`, the faster-whisper
shape) so the buffer/agreement logic is unit-testable without a model. Use
`WhisperStreamingASR.load(...)` to build the production transcribe_fn.
"""
from __future__ import annotations

import numpy as np

from .commit_policy import LocalAgreement


def _words_from_segments(segments) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            text = (w.word or "").strip()
            if not text:
                continue
            out.append((text, int((w.start or 0) * 1000), int((w.end or 0) * 1000)))
    return out


class WhisperStreamingASR:
    def __init__(self, transcribe_fn, sample_rate: int = 16000):
        self._transcribe = transcribe_fn
        self._sr = sample_rate
        self._buf = bytearray()
        self._policy = LocalAgreement()

    @classmethod
    def load(cls, model_dir, language: str, device: str = "cpu",
             compute_type: str = "int8", sample_rate: int = 16000):  # pragma: no cover - native
        from faster_whisper import WhisperModel
        model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)

        def _transcribe(audio):
            segments, _ = model.transcribe(
                audio, language=language, word_timestamps=True,
                beam_size=1, condition_on_previous_text=False, vad_filter=False,
            )
            return list(segments)

        return cls(_transcribe, sample_rate=sample_rate)

    def _decode(self):
        audio = np.frombuffer(bytes(self._buf), dtype=np.int16).astype(np.float32) / 32768.0
        segments = self._transcribe(audio)
        return _words_from_segments(segments)

    def feed(self, pcm_bytes: bytes):
        """Append PCM, re-decode, return (newly_committed, partial) word lists."""
        self._buf.extend(pcm_bytes)
        hypothesis = self._decode()
        committed = self._policy.add(hypothesis)
        return committed, self._policy.partial()

    def endpoint(self) -> list[tuple[str, int, int]]:
        """Commit whatever the latest hypothesis holds and reset for next segment."""
        remaining = self._policy.partial()
        final_words = list(self._policy.committed) + list(remaining)
        self._buf = bytearray()
        self._policy.reset()
        return final_words
