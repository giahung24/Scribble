"""Streaming utterance endpointer for pseudo-realtime local STT.

Consumes int16 mono PCM (16 kHz) incrementally and emits complete
utterances cut on trailing silence, with a hard max-duration cap and a
minimum-length filter that suppresses sub-second blips. Dependency-free
(stdlib only) so it stays fast and trivially unit-testable.

The default silence floor mirrors the upload splitter's -30 dBFS
convention (services/vad_splitter.py): 10**(-30/20) ≈ 0.0316 of full scale.
"""
from __future__ import annotations

import array
import math

_BYTES_PER_SAMPLE = 2  # int16


class StreamingEndpointer:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        silence_gap_ms: int = 600,
        min_utterance_ms: int = 1000,
        max_utterance_ms: int = 12000,
        silence_rms: float = 0.0316,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000))
        self.frame_ms = frame_ms
        self.silence_gap_frames = max(1, silence_gap_ms // frame_ms)
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_ms = max_utterance_ms
        self.silence_rms = silence_rms
        self._residual = b""            # bytes not yet aligned to a full frame
        self._utterance = bytearray()   # current utterance PCM
        self._in_speech = False
        self._trailing_silence_frames = 0

    def _frame_rms(self, frame: array.array) -> float:
        if not frame:
            return 0.0
        acc = 0.0
        for s in frame:
            x = s / 32768.0
            acc += x * x
        return math.sqrt(acc / len(frame))

    def _utterance_ms(self) -> int:
        samples = len(self._utterance) // _BYTES_PER_SAMPLE
        return int(samples * 1000 / self.sample_rate)

    def _close_utterance(self) -> bytes | None:
        utt = bytes(self._utterance)
        ms = self._utterance_ms()
        self._utterance = bytearray()
        self._in_speech = False
        self._trailing_silence_frames = 0
        if ms < self.min_utterance_ms:
            return None  # blip — suppress
        return utt

    def feed(self, pcm_bytes: bytes) -> list[bytes]:
        """Append PCM; return zero or more completed utterances (bytes)."""
        out: list[bytes] = []
        data = self._residual + pcm_bytes
        frame_bytes = self.frame_samples * _BYTES_PER_SAMPLE
        i = 0
        while i + frame_bytes <= len(data):
            frame_raw = data[i:i + frame_bytes]
            i += frame_bytes
            frame = array.array("h")
            frame.frombytes(frame_raw)
            is_speech = self._frame_rms(frame) >= self.silence_rms

            if is_speech:
                self._in_speech = True
                self._trailing_silence_frames = 0
                self._utterance.extend(frame_raw)
            elif self._in_speech:
                # keep trailing silence in the buffer — Whisper transcribes
                # complete utterances (with a little tail) more accurately
                self._utterance.extend(frame_raw)
                self._trailing_silence_frames += 1
                if self._trailing_silence_frames >= self.silence_gap_frames:
                    cut = self._close_utterance()
                    if cut is not None:
                        out.append(cut)
                    continue
            # else: leading silence before any speech — drop it

            if self._in_speech and self._utterance_ms() >= self.max_utterance_ms:
                cut = self._close_utterance()
                if cut is not None:
                    out.append(cut)

        self._residual = data[i:]
        return out

    def flush(self) -> bytes | None:
        """Close any pending utterance (call on stop). Returns it or None."""
        if self._in_speech and len(self._utterance) > 0:
            return self._close_utterance()
        return None
