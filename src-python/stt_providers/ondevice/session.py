"""On-device realtime session: drives WhisperStreamingASR + optional translation
on a worker thread, emitting the same result dicts as the other realtime
streamers (partial {is_final:False}; final {is_final:True, pcm}; the WS handler
adds diarization + translation delivery).

Endpoint detection is a simple trailing-silence RMS gate so a pause closes the
current segment and triggers a final + buffer reset.
"""
from __future__ import annotations

import array
import math
import queue
import threading

from logger import get_logger

log = get_logger(__name__)

_SR_BYTES = 2
_SENTINEL = object()


def _rms(frame: bytes) -> float:
    a = array.array("h"); a.frombytes(frame)
    if not a:
        return 0.0
    return math.sqrt(sum((s / 32768.0) ** 2 for s in a) / len(a))


class OnDeviceStreamingSession:
    def __init__(self, asr, translate=None, sample_rate: int = 16000,
                 silence_gap_ms: int = 600, silence_rms: float = 0.0316):
        self._asr = asr
        self._translate = translate          # callable(text)->str or None
        self._sr = sample_rate
        self._silence_frames_needed = max(1, silence_gap_ms // 30)
        self._frame_bytes = int(sample_rate * 30 / 1000) * _SR_BYTES
        self._silence_rms = silence_rms
        self._in_q: queue.Queue = queue.Queue(maxsize=256)
        self._out_q: queue.Queue = queue.Queue()
        self._worker = None
        self._stopped = False

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._stopped = False
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def feed_audio(self, pcm_bytes: bytes):
        if self._stopped:
            return
        try:
            self._in_q.put_nowait(pcm_bytes)
        except queue.Full:
            pass

    def stop(self):
        if self._worker is None:
            return
        self._stopped = True
        try:
            self._in_q.put(_SENTINEL, timeout=5.0)
        except queue.Full:
            self._out_q.put(None)

    def results(self):
        while True:
            item = self._out_q.get()
            if item is None:
                return
            yield item

    def _emit_final(self, words, seg_pcm):
        text = " ".join(w[0] for w in words).strip()
        if not text:
            return
        msg = {"text": text, "is_final": True, "pcm": seg_pcm}
        if self._translate:
            try:
                tl = self._translate(text)
                if tl:
                    msg["translation"] = tl
            except Exception:
                pass
        self._out_q.put(msg)

    def _loop(self):
        trailing_silence = 0
        seg_pcm = bytearray()
        residual = b""
        try:
            while True:
                try:
                    item = self._in_q.get(timeout=0.5)
                except queue.Empty:
                    if self._stopped:
                        break
                    continue
                if item is _SENTINEL:
                    break
                try:
                    seg_pcm.extend(item)
                    committed, partial = self._asr.feed(item)
                    if partial or committed:
                        preview = " ".join(w[0] for w in (list(committed) + list(partial))).strip()
                        if preview:
                            self._out_q.put({"text": preview, "is_final": False})
                    data = residual + item
                    i = 0
                    while i + self._frame_bytes <= len(data):
                        frame = data[i:i + self._frame_bytes]; i += self._frame_bytes
                        if _rms(frame) < self._silence_rms:
                            trailing_silence += 1
                        else:
                            trailing_silence = 0
                    residual = data[i:]
                    if trailing_silence >= self._silence_frames_needed:
                        self._emit_final(self._asr.endpoint(), bytes(seg_pcm))
                        seg_pcm = bytearray()
                        trailing_silence = 0
                except Exception:  # noqa: BLE001 — one bad chunk must not kill the session
                    log.warning("[ondevice] worker chunk error", exc_info=True)
                    continue
            # flush any remaining segment on shutdown (sentinel or _stopped)
            try:
                self._emit_final(self._asr.endpoint(), bytes(seg_pcm))
            except Exception:  # noqa: BLE001
                log.warning("[ondevice] endpoint flush error", exc_info=True)
        finally:
            self._out_q.put(None)  # ALWAYS terminate results(), even on fatal error
