# Local Realtime STT (v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pseudo-realtime live transcription (+ live translation) to the local OpenAI-compatible Whisper provider by buffering live PCM, cutting utterances on silence, and transcribing each through the existing batch endpoint.

**Architecture:** A new streaming endpointer cuts the live PCM stream into utterance-sized chunks on speech pauses. A `LocalRealtimeSession` runs a sequential worker that transcribes each utterance via the v1 `LocalOpenAIProvider.transcribe_file` and yields finals. A new `/ws/local-stream` WebSocket handler — structurally parallel to the existing `nvidia_stream_ws` — drives the session, applies CAM++ diarization to the utterance PCM, runs per-chunk `translate_instant`, and emits the existing realtime JSON message shapes. The frontend routes `provider === "local"` to the new path.

**Tech Stack:** Python 3.10 / FastAPI (sidecar), `httpx` (provider), `numpy` (diarization), React 19 / TypeScript (frontend). Tests: pytest.

---

## File Structure

**Create:**
- `src-python/stt_providers/realtime_endpointer.py` — `StreamingEndpointer`: dependency-free RMS endpointer over int16 PCM. One responsibility: turn a PCM stream into completed utterances.
- `src-python/stt_providers/local_realtime.py` — `LocalRealtimeSession`: buffer + worker + temp-WAV + provider call; yields result dicts. One responsibility: pseudo-realtime session lifecycle.
- `src-python/tests/test_realtime_endpointer.py` — endpointer unit tests.
- `src-python/tests/test_local_realtime.py` — session unit tests (mocked provider).

**Modify:**
- `src-python/stt_providers/local_openai.py` — flip `supports_realtime = True`; add `open_session()` returning a `LocalRealtimeSession`.
- `src-python/main.py` — add `@app.websocket("/ws/local-stream")` handler.
- `src/components/recording/recording-constants.ts` — add `WS_PATH_LOCAL`.
- `src/components/recording/use-streaming-stt.ts` — route `provider === "local"`.
- `src/components/RecordingBar.tsx` — add `local` readiness branch (Base URL + Model).
- `README.md` — note realtime support for the local provider.

**Test command (Python):** `src-python/.venv/bin/python -m pytest src-python/tests/ -v`
**Type-check (frontend):** `npx tsc --noEmit`

---

## Task 1: Streaming utterance endpointer

**Files:**
- Create: `src-python/stt_providers/realtime_endpointer.py`
- Test: `src-python/tests/test_realtime_endpointer.py`

- [ ] **Step 1: Write the failing tests**

Create `src-python/tests/test_realtime_endpointer.py`:

```python
"""Unit tests for the streaming utterance endpointer."""
from stt_providers.realtime_endpointer import StreamingEndpointer

SR = 16000


def _pcm(ms: int, amplitude: int) -> bytes:
    """Build `ms` milliseconds of constant-amplitude int16 mono PCM."""
    import array
    n = int(SR * ms / 1000)
    return array.array("h", [amplitude] * n).tobytes()


def _speech(ms: int) -> bytes:
    # amplitude 8000 → RMS ≈ 0.244 (well above the ~0.0316 silence floor)
    return _pcm(ms, 8000)


def _silence(ms: int) -> bytes:
    return _pcm(ms, 0)


def test_speech_then_silence_emits_one_utterance():
    ep = StreamingEndpointer(sample_rate=SR)
    out = ep.feed(_speech(1500) + _silence(700))
    assert len(out) == 1
    # utterance includes the trailing silence that closed it
    samples = len(out[0]) // 2
    assert samples >= SR * 1.4  # ~1.5s of speech retained


def test_short_blip_is_suppressed():
    ep = StreamingEndpointer(sample_rate=SR, min_utterance_ms=1000)
    out = ep.feed(_speech(300) + _silence(700))
    assert out == []
    assert ep.flush() is None


def test_continuous_speech_force_flushes_at_cap():
    ep = StreamingEndpointer(sample_rate=SR, max_utterance_ms=12000)
    out = ep.feed(_speech(13000))  # no pause → only the hard cap can cut
    assert len(out) >= 1
    first_ms = (len(out[0]) // 2) * 1000 // SR
    assert first_ms >= 11000  # forced cut landed near the 12s cap


def test_flush_returns_pending_tail():
    ep = StreamingEndpointer(sample_rate=SR, min_utterance_ms=500)
    out = ep.feed(_speech(1200))  # no trailing silence → nothing cut yet
    assert out == []
    tail = ep.flush()
    assert tail is not None
    assert (len(tail) // 2) >= SR  # ~1.2s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_realtime_endpointer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stt_providers.realtime_endpointer'`

- [ ] **Step 3: Implement the endpointer**

Create `src-python/stt_providers/realtime_endpointer.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_realtime_endpointer.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/realtime_endpointer.py src-python/tests/test_realtime_endpointer.py
git commit -m "feat(stt): streaming utterance endpointer for local realtime"
```

---

## Task 2: LocalRealtimeSession

**Files:**
- Create: `src-python/stt_providers/local_realtime.py`
- Test: `src-python/tests/test_local_realtime.py`

- [ ] **Step 1: Write the failing tests**

Create `src-python/tests/test_local_realtime.py`:

```python
"""Unit tests for LocalRealtimeSession (mocked provider, no network)."""
import array

from stt_providers.base import STTSegment
from stt_providers.local_realtime import LocalRealtimeSession

SR = 16000


def _speech(ms: int) -> bytes:
    n = int(SR * ms / 1000)
    return array.array("h", [8000] * n).tobytes()


def _silence(ms: int) -> bytes:
    n = int(SR * ms / 1000)
    return array.array("h", [0] * n).tobytes()


class _FakeProvider:
    """Stand-in for LocalOpenAIProvider — records calls, returns canned text."""
    def __init__(self):
        self.calls = []

    def transcribe_file(self, wav_path, language, **opts):
        self.calls.append((str(wav_path), language))
        return [STTSegment(text="hello world")]


def test_session_yields_interim_then_final_with_pcm():
    provider = _FakeProvider()
    session = LocalRealtimeSession(provider, language="en", sample_rate=SR)
    session.start()
    # one utterance: 1.5s speech + 0.7s trailing silence
    session.feed_audio(_speech(1500) + _silence(700))
    session.stop()

    results = list(session.results())
    interims = [r for r in results if not r["is_final"]]
    finals = [r for r in results if r["is_final"]]

    assert len(interims) >= 1                       # activity tick emitted
    assert len(finals) == 1
    assert finals[0]["text"] == "hello world"
    assert isinstance(finals[0]["pcm"], (bytes, bytearray))  # PCM passed for diarization
    assert len(provider.calls) == 1
    assert provider.calls[0][1] == "en"  # transcribed once, with the session language


def test_session_transcribes_trailing_tail_on_stop():
    provider = _FakeProvider()
    session = LocalRealtimeSession(provider, language="vi", sample_rate=SR)
    session.start()
    session.feed_audio(_speech(1300))  # no trailing silence → only flush() can cut it
    session.stop()                     # stop() must flush the pending tail

    finals = [r for r in session.results() if r["is_final"]]
    assert len(finals) == 1
    assert finals[0]["text"] == "hello world"


def test_session_skips_empty_transcription():
    class _EmptyProvider:
        def transcribe_file(self, wav_path, language, **opts):
            return []  # server returned nothing usable
    session = LocalRealtimeSession(_EmptyProvider(), language="en", sample_rate=SR)
    session.start()
    session.feed_audio(_speech(1500) + _silence(700))
    session.stop()
    finals = [r for r in session.results() if r["is_final"]]
    assert finals == []  # no final emitted for empty text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_local_realtime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stt_providers.local_realtime'`

- [ ] **Step 3: Implement the session**

Create `src-python/stt_providers/local_realtime.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_local_realtime.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/local_realtime.py src-python/tests/test_local_realtime.py
git commit -m "feat(stt): LocalRealtimeSession — VAD-chunked batch streaming"
```

---

## Task 3: Wire provider capability + open_session

**Files:**
- Modify: `src-python/stt_providers/local_openai.py:62-74`
- Test: `src-python/tests/test_local_openai.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `src-python/tests/test_local_openai.py`:

```python
def test_provider_supports_realtime_and_opens_session():
    from stt_providers.local_openai import LocalOpenAIProvider
    from stt_providers.local_realtime import LocalRealtimeSession
    p = LocalOpenAIProvider("http://localhost:8000", "whisper-1")
    assert p.supports_realtime is True
    session = p.open_session("en")
    assert isinstance(session, LocalRealtimeSession)
    assert session._language == "en"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_local_openai.py::test_provider_supports_realtime_and_opens_session -v`
Expected: FAIL — `assert False is True` (supports_realtime currently False)

- [ ] **Step 3: Implement**

In `src-python/stt_providers/local_openai.py`, change the class attribute (currently `supports_realtime = False   # v2`):

```python
    supports_realtime = True    # v2: pseudo-realtime via VAD-chunked batch
```

Then add this method to `LocalOpenAIProvider` (after `transcribe_file`):

```python
    def open_session(self, language: str, **opts):
        """Realtime session — buffers live PCM and transcribes utterances
        through this same provider's batch endpoint. See LocalRealtimeSession."""
        from .local_realtime import LocalRealtimeSession
        return LocalRealtimeSession(self, language)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/ -v`
Expected: PASS (all tests, including the new one — 28+ total)

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/local_openai.py src-python/tests/test_local_openai.py
git commit -m "feat(stt): local provider opens realtime sessions"
```

---

## Task 4: `/ws/local-stream` WebSocket handler

**Files:**
- Modify: `src-python/main.py` (add handler after the Soniox handler block)

> No unit test — the WS handler is verified by manual E2E, consistent with how
> `nvidia_stream_ws` and `soniox_stream_ws` are tested (neither has unit tests).
> Step 4 below is a syntax/compile gate plus a manual E2E checklist.

- [ ] **Step 1: Add the handler**

Add the following to `src-python/main.py` as a **new module-level function** (same indentation as the other `@app.websocket(...)` handlers — NOT nested inside another function). Locate the end of `soniox_stream_ws` with `grep -n 'def soniox_stream_ws' src-python/main.py`, then insert after that function's body (the next top-level `@app...` or `def` marks the boundary):

```python
# ─── Local (OpenAI-compatible Whisper) Pseudo-Realtime WebSocket ───
@app.websocket("/ws/local-stream")
async def local_stream_ws(websocket: WebSocket):
    """Pseudo-realtime local STT: buffer live PCM, cut utterances on silence,
    transcribe each via the batch /v1/audio/transcriptions endpoint, diarize
    with CAM++, translate per chunk. Mirrors nvidia_stream_ws delivery shapes."""
    await websocket.accept()

    diarizer.reset()
    source = websocket.query_params.get("source", "web")
    diarizer.set_source(source)
    max_sp = db.get_setting("max_speakers")
    if max_sp:
        try:
            diarizer.set_max_speakers(int(max_sp))
        except (ValueError, TypeError):
            pass

    meeting_id_raw = websocket.query_params.get("meeting_id")
    archive_fh = None
    if meeting_id_raw:
        try:
            meeting_id = int(meeting_id_raw)
            meeting = db.get_meeting(meeting_id)
            if meeting:
                audio_dir = _voicescribe_data_dir() / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                archive_path = audio_dir / f"meeting_{meeting_id}.pcm"
                archive_fh = archive_path.open("ab")
                db.update_meeting(meeting_id, audio_path=str(archive_path))
        except Exception as e:
            log.warning("[ws:local-stream] archive setup failed: %s", e)

    stt_lang = db.get_setting("stt_language") or "vi"
    translation_tasks = set()
    # Defined in the OUTER scope so both _send_results and the receive loop's
    # TRANSLATE command share it via closure.
    translate_state = {"lang": websocket.query_params.get("translate_lang", "")}

    def _close_archive():
        nonlocal archive_fh
        if archive_fh is not None:
            try:
                archive_fh.close()
            except Exception:
                pass
            archive_fh = None

    # Build the provider (validates Base URL + Model; raises if unset).
    from stt_providers.registry import build_local_provider
    try:
        provider = build_local_provider(db)
    except Exception as e:
        _close_archive()
        await websocket.send_json({
            "error": True, "terminal": True, "text": str(e),
            "is_final": True, "speaker": "System", "speaker_id": -1,
        })
        await websocket.close()
        return

    session = provider.open_session(stt_lang)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, session.start)
    except Exception as e:
        _close_archive()
        await websocket.send_json({
            "error": True, "terminal": True, "text": str(e),
            "is_final": True, "speaker": "System", "speaker_id": -1,
        })
        await websocket.close()
        return

    result_queue = asyncio.Queue()

    def _read_results():
        for result in session.results():
            asyncio.run_coroutine_threadsafe(result_queue.put(result), loop)
        asyncio.run_coroutine_threadsafe(result_queue.put(None), loop)

    result_thread = threading.Thread(target=_read_results, daemon=True)
    result_thread.start()

    DIARIZE_MIN_BYTES = 16000  # 0.5s at 16kHz int16

    async def _send_results():
        from translate import translate_instant
        current_chunk_id = f"chunk-{int(time.time() * 1000)}-{uuid4().hex[:8]}"

        transcript_parts: list[dict] = []
        last_save_at = time.time()
        SAVE_INTERVAL = 10.0

        def _accumulate_part(text, speaker, speaker_id, chunk_id):
            # Simpler than the Nvidia handler's _accumulate_part on purpose: the
            # Nvidia path carries chunkData/chunkIds to REPLACE in-progress interim
            # text in place. Local emits finals only (no interim text to replace),
            # so a part needs just {text, speaker, speakerId, chunkId}. This is a
            # subset of the Nvidia persisted shape and renders identically on
            # reopen — TranscriptPart treats chunkIds as optional, and the
            # translation matcher checks `chunkId === chunk_id` as well as chunkIds.
            if not text.strip():
                return
            if transcript_parts and transcript_parts[-1].get("speakerId") == speaker_id:
                p = transcript_parts[-1]
                p["text"] = (p["text"] + " " + text).strip()
            else:
                transcript_parts.append({
                    "text": text, "speaker": speaker,
                    "speakerId": speaker_id, "chunkId": chunk_id,
                })

        def _flush_to_db():
            nonlocal last_save_at
            if not meeting_id_raw or not transcript_parts:
                return
            try:
                db.update_meeting(int(meeting_id_raw),
                                  transcript=json.dumps(transcript_parts, ensure_ascii=False))
                last_save_at = time.time()
            except Exception as e:
                log.warning("[ws:local auto-save] error: %s", e)

        last_speaker = "Speaker 1"
        last_speaker_id = 0

        while True:
            result = await result_queue.get()
            if result is None:
                break
            try:
                # Interim activity tick — keep the UI's live area from looking frozen
                if not result.get("is_final"):
                    await websocket.send_json({
                        "text": result.get("text", "…"), "is_final": False,
                        "speaker": last_speaker, "speaker_id": last_speaker_id,
                        "chunk_id": current_chunk_id,
                    })
                    continue

                text = result.get("text", "")
                if not text.strip():
                    continue

                msg = {
                    "text": text, "is_final": True,
                    "speaker": last_speaker, "speaker_id": last_speaker_id,
                    "chunk_id": current_chunk_id,
                }

                # Diarize from the exact utterance PCM the session transcribed
                pcm = result.get("pcm")
                if pcm and len(pcm) >= DIARIZE_MIN_BYTES:
                    try:
                        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                        speaker_info = await loop.run_in_executor(
                            None, diarizer.identify_speaker_from_samples, samples, 16000
                        )
                        msg["speaker"] = speaker_info.get("speaker", last_speaker)
                        msg["speaker_id"] = speaker_info.get("speaker_id", last_speaker_id)
                        last_speaker = msg["speaker"]
                        last_speaker_id = msg["speaker_id"]
                    except Exception as e:
                        log.warning("[ws:local] diarize error: %s", e)

                await websocket.send_json(msg)
                _accumulate_part(msg["text"], msg["speaker"], msg["speaker_id"], current_chunk_id)

                # Per-chunk translation (capture the chunk_id BEFORE advancing it)
                if translate_state["lang"]:
                    _text = msg["text"]
                    _cid = msg["chunk_id"]
                    _lang = translate_state["lang"]
                    _src = stt_lang

                    async def _do_translate(text=_text, cid=_cid, lang=_lang, src=_src):
                        import re
                        if not text or not lang:
                            return
                        if not re.sub(r'[^\w\s]', '', text).strip():
                            return
                        try:
                            translated = await loop.run_in_executor(
                                None, translate_instant, text, lang, db, src
                            )
                            if translated:
                                await websocket.send_json({
                                    "type": "translation",
                                    "translation": translated,
                                    "chunk_id": cid,
                                    "append": True,
                                })
                        except Exception as e:
                            log.warning("[ws:local-trans] error: %s", e)

                    t = asyncio.create_task(_do_translate())
                    translation_tasks.add(t)
                    t.add_done_callback(translation_tasks.discard)

                current_chunk_id = f"chunk-{int(time.time() * 1000)}-{uuid4().hex[:8]}"

                if time.time() - last_save_at >= SAVE_INTERVAL:
                    _flush_to_db()
            except WebSocketDisconnect:
                break

        _flush_to_db()

    send_task = asyncio.create_task(_send_results())

    try:
        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                break
            if "bytes" in data:
                audio_bytes = data["bytes"]
                session.feed_audio(audio_bytes)
                if archive_fh is not None:
                    try:
                        archive_fh.write(audio_bytes)
                    except Exception as e:
                        log.warning("[ws:local-stream] archive write failed: %s", e)
            elif "text" in data:
                txt = data["text"]
                if txt == "STOP":
                    break
                if txt.startswith("TRANSLATE:"):
                    lang_cmd = txt[len("TRANSLATE:"):].strip()
                    translate_state["lang"] = "" if lang_cmd.lower() == "off" else lang_cmd
                    log.info("[ws:local] translation set mid-session: '%s'", translate_state["lang"])
    except WebSocketDisconnect:
        pass
    finally:
        session.stop()
        result_thread.join(timeout=5)
        try:
            await asyncio.wait_for(send_task, timeout=3.0)
        except asyncio.TimeoutError:
            send_task.cancel()
        if translation_tasks:
            await asyncio.wait(translation_tasks, timeout=5.0)
        _close_archive()
        try:
            await websocket.close()
        except Exception:
            pass
```

- [ ] **Step 2: Verify the file compiles**

Run: `src-python/.venv/bin/python -m py_compile src-python/main.py`
Expected: no output, exit 0 (syntax valid)

- [ ] **Step 3: Verify the route is registered**

Run: `grep -n 'ws/local-stream' src-python/main.py`
Expected: one match on the `@app.websocket(...)` decorator line.

- [ ] **Step 4: Run the full Python suite (no regressions)**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/ -v`
Expected: PASS (all tests still green)

- [ ] **Step 5: Commit**

```bash
git add src-python/main.py
git commit -m "feat(stt): /ws/local-stream pseudo-realtime handler"
```

---

## Task 5: Frontend WebSocket routing

**Files:**
- Modify: `src/components/recording/recording-constants.ts:13`
- Modify: `src/components/recording/use-streaming-stt.ts:5-11,21`

- [ ] **Step 1: Add the path constant**

In `src/components/recording/recording-constants.ts`, after the `WS_PATH_SONIOX` line, add:

```typescript
export const WS_PATH_LOCAL = "/ws/local-stream";
```

- [ ] **Step 2: Import and route it**

In `src/components/recording/use-streaming-stt.ts`, add `WS_PATH_LOCAL` to the import block from `./recording-constants`:

```typescript
import {
    WS_CONNECT_TIMEOUT_MS,
    WS_PATH_NVIDIA,
    WS_PATH_SONIOX,
    WS_PATH_LOCAL,
    TARGET_SAMPLE_RATE,
    SCRIPT_PROCESSOR_BUFFER,
} from "./recording-constants";
```

Then replace the `wsPath` assignment in `openStreamingWebSocket` (currently
`const wsPath = provider === "soniox" ? WS_PATH_SONIOX : WS_PATH_NVIDIA;`) with:

```typescript
    const wsPath =
        provider === "soniox" ? WS_PATH_SONIOX
        : provider === "local" ? WS_PATH_LOCAL
        : WS_PATH_NVIDIA;
```

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: `TypeScript: No errors found`

- [ ] **Step 4: Commit**

```bash
git add src/components/recording/recording-constants.ts src/components/recording/use-streaming-stt.ts
git commit -m "feat(recording): route local provider to /ws/local-stream"
```

---

## Task 6: Record-start readiness for local

**Files:**
- Modify: `src/components/RecordingBar.tsx:358-367`

> The existing gate checks `provider === 'nvidia'` and `provider === 'soniox'`
> explicitly, so `local` already falls through without being wrongly blocked
> (unlike the upload gate we fixed earlier). This task adds a `local` branch so
> a misconfigured local provider gives a helpful message instead of failing only
> after the WebSocket opens.

- [ ] **Step 1: Add the local readiness branch**

In `src/components/RecordingBar.tsx`, immediately after the closing `}` of the
`if (provider === 'soniox' && !sonioxConfigured) { ... }` block (line ~367) and
before the `} catch (e) {`, insert:

```typescript
                if (provider === 'local') {
                    const baseUrl = (settings.local_stt_base_url || '').trim();
                    const model = (settings.local_stt_model || '').trim();
                    if (!baseUrl || !model) {
                        showToast(
                            lang === 'vi'
                                ? 'Vui lòng cấu hình Local STT (Base URL + Model) trong Cài đặt trước khi ghi âm'
                                : 'Please configure Local STT (Base URL + Model) in Settings before recording',
                            'error'
                        );
                        useAppStore.getState().setSettingsOpen(true);
                        return;
                    }
                }
```

- [ ] **Step 2: Type-check**

Run: `npx tsc --noEmit`
Expected: `TypeScript: No errors found`

- [ ] **Step 3: Commit**

```bash
git add src/components/RecordingBar.tsx
git commit -m "feat(recording): local STT readiness check before recording"
```

---

## Task 7: Document realtime support

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the local STT description**

In `README.md`, find the Local (Whisper) STT section added in v1 (search:
`grep -n -i 'local.*whisper' README.md`). Add a sentence noting realtime is now
supported. Append to that section:

```markdown
> **Realtime (v2):** The local provider now supports live recording in addition
> to upload. It is pseudo-realtime — audio is cut into utterances on speech
> pauses and transcribed through the same `/v1/audio/transcriptions` endpoint, so
> segments appear after each utterance (2–6 s typical) rather than word-by-word.
> Use a GPU or a smaller/distilled model for the lowest latency. Live translation
> is supported via the existing translation setting.
```

- [ ] **Step 2: Verify the edit**

Run: `grep -n 'Realtime (v2)' README.md`
Expected: one match.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: local STT realtime (v2) support"
```

---

## Final verification (after all tasks)

- [ ] Run full Python suite: `src-python/.venv/bin/python -m pytest src-python/tests/ -v` → all pass.
- [ ] Type-check frontend: `npx tsc --noEmit` → no errors.
- [ ] `python -m py_compile src-python/main.py` → exit 0.
- [ ] Manual E2E (requires a running OpenAI-compatible server, e.g. Speaches at `http://localhost:8000`):
  1. Settings → STT → Local; set Base URL + Model; Provider = Local.
  2. Press Record, speak a few sentences in VI / EN / FR with pauses.
  3. Verify: segments appear after each pause; speaker labels populate (CAM++);
     enabling translation shows translated text per segment; STOP ends cleanly.
  4. Misconfigure (clear Model) → pressing Record shows the readiness toast.
  5. Point Base URL at a non-OpenAI server → connect yields a terminal error
     toast and recording stops (no hang).
```
