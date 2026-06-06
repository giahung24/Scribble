"""Pseudo-realtime local STT session.

Buffers live int16 PCM, cuts utterances with StreamingEndpointer, and runs a
single sequential worker that transcribes each utterance through the batch
provider (LocalOpenAIProvider.transcribe_file). Yields result dicts shaped
like the other realtime streamers (NvidiaStreamingSTT): an interim activity
tick {"text": "…", "is_final": False} when a chunk is cut, then a final
{"text": ..., "is_final": True, "pcm": <bytes>} when transcription completes.
The handler uses the attached PCM for CAM++ diarization.
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading
import wave

from logger import get_logger
from .base import segments_to_text
from .realtime_endpointer import StreamingEndpointer

log = get_logger(__name__)

# Generous: utterances arrive only every few seconds, so this bounds memory
# without ever filling in practice. If a catastrophically slow server fills it,
# we drop + log rather than block the audio receive loop.
_MAX_PENDING_UTTERANCES = 64
_INTERIM_TICK = {"text": "…", "is_final": False}
_SENTINEL = object()


class LocalRealtimeSession:
    def __init__(self, provider, language: str, sample_rate: int = 16000):
        self._provider = provider
        self._language = language
        self._sr = sample_rate
        self._endpointer = StreamingEndpointer(sample_rate=sample_rate)
        self._chunk_q: queue.Queue = queue.Queue(maxsize=_MAX_PENDING_UTTERANCES)
        self._out_q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stopped = False

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        self._stopped = False
        self._worker = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        # Stop accepting audio, flush any pending tail, then signal end-of-work.
        self._stopped = True
        tail = self._endpointer.flush()
        if tail:
            self._enqueue(tail)
        try:
            self._chunk_q.put_nowait(_SENTINEL)
        except queue.Full:
            # queue saturated — drain one and retry so the sentinel always lands
            try:
                self._chunk_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._chunk_q.put_nowait(_SENTINEL)
            except queue.Full:
                pass

    # ── feed ───────────────────────────────────────────────────────────────
    def feed_audio(self, pcm_bytes: bytes) -> None:
        if self._stopped:
            return
        for utt in self._endpointer.feed(pcm_bytes):
            # emit the activity tick immediately, then queue for transcription
            self._out_q.put(dict(_INTERIM_TICK))
            self._enqueue(utt)

    def _enqueue(self, utt: bytes) -> None:
        try:
            self._chunk_q.put_nowait(utt)
        except queue.Full:
            log.warning("[local-rt] transcribe backlog full — dropping utterance "
                        "(server slower than realtime; use a faster model/GPU)")

    # ── results ──────────────────────────────────────────────────────────────
    def results(self):
        while True:
            item = self._out_q.get()
            if item is None:
                return
            yield item

    # ── worker ─────────────────────────────────────────────────────────────
    def _transcribe_loop(self) -> None:
        while True:
            utt = self._chunk_q.get()
            if utt is _SENTINEL:
                break
            try:
                text = self._transcribe(utt)
            except Exception as e:  # noqa: BLE001 — one bad chunk must not kill the session
                log.warning("[local-rt] chunk transcription failed: %s", e)
                continue
            if text and text.strip():
                self._out_q.put({"text": text, "is_final": True, "pcm": utt})
        self._out_q.put(None)  # end the results() iterator

    def _transcribe(self, pcm: bytes) -> str:
        path = self._write_wav(pcm)
        try:
            segments = self._provider.transcribe_file(path, self._language)
            return segments_to_text(segments)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _write_wav(self, pcm: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="local-rt-")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sr)
            wf.writeframes(pcm)
        return path
