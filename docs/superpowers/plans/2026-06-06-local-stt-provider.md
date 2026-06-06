# Local STT Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, self-hosted, OpenAI-compatible transcription provider (Whisper by default) to Scribble's batch Upload pipeline, alongside Nvidia Riva and Soniox.

**Architecture:** A new `src-python/stt_providers/` package defines a normalized `STTSegment` + `STTProvider` interface and a `LocalOpenAIProvider` that POSTs audio chunks to `{base_url}/v1/audio/transcriptions`. The batch pipeline (`upload_pipeline.py`) resolves a per-chunk transcribe callable instead of hardcoding Nvidia/Soniox, so the local provider reuses the existing VAD-split → per-chunk transcribe → CAM++ clustering → resume framework unchanged. Frontend gains a third STT provider tab with Base URL + Model + optional API key (mirroring the existing OpenAI-compatible LLM config).

**Tech Stack:** Python 3.10+ / FastAPI / httpx (sync), pytest; React 19 / TypeScript; SQLite key/value settings.

**Scope:** v1 = batch (Upload) only. Realtime + cabin translation are explicitly deferred (see spec §8). Do NOT touch the realtime WS endpoints or Riva/Soniox internals.

**Reference spec:** `docs/superpowers/specs/2026-06-06-local-stt-provider-design.md`

---

## File Structure

**New files:**
- `src-python/stt_providers/__init__.py` — package marker + public exports
- `src-python/stt_providers/base.py` — `STTSegment` dataclass, `STTProvider` ABC, `segments_to_text()`
- `src-python/stt_providers/local_openai.py` — `LocalOpenAIProvider` + `_parse_transcription_response()`
- `src-python/stt_providers/registry.py` — `build_local_provider(db)`, `build_local_transcriber(db)`
- `src-python/tests/conftest.py` — puts `src-python/` on `sys.path` for imports
- `src-python/tests/test_local_openai.py` — provider parsing + transcribe tests
- `src-python/tests/test_registry.py` — local transcriber/provider construction + validation

**Modified files:**
- `src-python/requirements.txt` — add pytest deps
- `src-python/services/upload_pipeline.py` — extract `_resolve_chunk_transcriber()`, add `local` branch
- `src-python/api/settings.py` — mask `local_stt_api_key`, add `GET /stt-models`
- `src/lib/api.ts` — add `fetchSttModels()`
- `src/components/SettingsPanel.tsx` — add `local` provider tab + config block
- `README.md` — document the local STT backend

**Unchanged:** realtime WS endpoints (`main.py`), Riva/Soniox internals (`stt.py`, `translate.py`), DB schema, Tauri/Rust layer.

---

## Task 1: Python test infrastructure

No pytest setup exists in `src-python/`. Establish it first so later tasks are TDD.

**Files:**
- Modify: `src-python/requirements.txt`
- Create: `src-python/tests/conftest.py`
- Create: `src-python/tests/__init__.py` (empty)

- [ ] **Step 1: Add pytest deps to requirements.txt**

Append these lines to `src-python/requirements.txt` (after `pyinstaller>=6.0`):

```
# ── Dev/test only (not bundled by PyInstaller) ──
pytest>=8.0
pytest-asyncio>=0.23
```

- [ ] **Step 2: Create the test package + conftest**

The sidecar uses flat imports (`from db import Database`, `from stt import ...`). Tests must run with `src-python/` on `sys.path`.

Create `src-python/tests/__init__.py` as an empty file.

Create `src-python/tests/conftest.py`:

```python
"""Pytest bootstrap: put the sidecar root (src-python/) on sys.path so tests
can use the same flat imports the app uses (e.g. `from stt_providers...`)."""
import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))
```

- [ ] **Step 3: Add a smoke test to prove pytest runs**

Create `src-python/tests/test_smoke.py`:

```python
def test_pytest_runs():
    assert True
```

- [ ] **Step 4: Install deps and run**

Run (from `src-python/`, in the project venv):
```bash
cd src-python && pip install -r requirements.txt && python -m pytest tests/test_smoke.py -v
```
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src-python/requirements.txt src-python/tests/
git commit -m "test(stt): add pytest infrastructure for sidecar"
```

---

## Task 2: STTSegment + STTProvider base

**Files:**
- Create: `src-python/stt_providers/__init__.py`
- Create: `src-python/stt_providers/base.py`
- Test: `src-python/tests/test_local_openai.py` (created here, expanded in Task 3)

- [ ] **Step 1: Write the failing test**

Create `src-python/tests/test_local_openai.py`:

```python
from stt_providers.base import STTSegment, segments_to_text


def test_segment_defaults():
    seg = STTSegment(text="hello", start_ms=0, end_ms=500)
    assert seg.text == "hello"
    assert seg.speaker_id is None
    assert seg.is_final is True
    assert seg.language is None


def test_segments_to_text_joins_and_strips():
    segs = [
        STTSegment(text="  xin chào ", start_ms=0, end_ms=10),
        STTSegment(text="thế giới  ", start_ms=10, end_ms=20),
    ]
    assert segments_to_text(segs) == "xin chào thế giới"


def test_segments_to_text_empty():
    assert segments_to_text([]) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src-python && python -m pytest tests/test_local_openai.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stt_providers'`.

- [ ] **Step 3: Write the implementation**

Create `src-python/stt_providers/__init__.py`:

```python
"""Pluggable STT providers. v1 exposes a local OpenAI-compatible provider for
the batch Upload pipeline; the interface is shaped to extend to realtime in v2."""
from .base import STTSegment, STTProvider, segments_to_text
from .local_openai import LocalOpenAIProvider

__all__ = ["STTSegment", "STTProvider", "segments_to_text", "LocalOpenAIProvider"]
```

Create `src-python/stt_providers/base.py`:

```python
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
    start_ms: int | None
    end_ms: int | None
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
```

> Note: `__init__.py` imports `LocalOpenAIProvider` which doesn't exist until Task 3. To keep this task's tests green, temporarily comment the `local_openai` import line in `__init__.py`, OR run the test against `stt_providers.base` directly (the test imports `from stt_providers.base import ...`, so the package `__init__` is still executed). **Do Task 3 immediately after** so `__init__.py` resolves. If you want this task fully green in isolation, create a stub `local_openai.py` now: see Task 3 Step 3 — you may create that file first.

- [ ] **Step 4: Make the package import cleanly**

Create a minimal `src-python/stt_providers/local_openai.py` stub so `__init__.py` imports succeed (full impl in Task 3):

```python
from .base import STTProvider

class LocalOpenAIProvider(STTProvider):
    name = "local"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd src-python && python -m pytest tests/test_local_openai.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src-python/stt_providers/ src-python/tests/test_local_openai.py
git commit -m "feat(stt): add STTSegment + STTProvider interface"
```

---

## Task 3: LocalOpenAIProvider (HTTP transcription + response parsing)

**Files:**
- Modify: `src-python/stt_providers/local_openai.py`
- Test: `src-python/tests/test_local_openai.py`

- [ ] **Step 1: Write the failing tests**

Append to `src-python/tests/test_local_openai.py`:

```python
import httpx
import pytest
from stt_providers.local_openai import LocalOpenAIProvider, _parse_transcription_response


def test_parse_verbose_json_segments():
    payload = {
        "language": "vietnamese",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": " Xin chào"},
            {"start": 1.5, "end": 3.0, "text": "mọi người "},
        ],
    }
    segs = _parse_transcription_response(payload)
    assert [s.text for s in segs] == ["Xin chào", "mọi người"]
    assert segs[0].start_ms == 0 and segs[0].end_ms == 1500
    assert segs[1].start_ms == 1500 and segs[1].end_ms == 3000
    assert segs[0].language == "vietnamese"


def test_parse_plain_text_fallback():
    segs = _parse_transcription_response({"text": "hello world"})
    assert len(segs) == 1
    assert segs[0].text == "hello world"
    assert segs[0].start_ms is None and segs[0].end_ms is None


def test_parse_empty():
    assert _parse_transcription_response({"text": "   "}) == []
    assert _parse_transcription_response({"segments": []}) == []


def test_transcribe_file_posts_and_parses(tmp_path, monkeypatch):
    wav = tmp_path / "chunk.wav"
    wav.write_bytes(b"RIFFfake")
    captured = {}

    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"segments": [{"start": 0, "end": 1, "text": "ok"}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["data"] = kwargs.get("data")
        captured["headers"] = kwargs.get("headers")
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = LocalOpenAIProvider("http://localhost:9000/", "whisper-1", api_key="secret")
    segs = provider.transcribe_file(wav, "vi")

    assert captured["url"] == "http://localhost:9000/v1/audio/transcriptions"
    assert captured["data"]["model"] == "whisper-1"
    assert captured["data"]["language"] == "vi"
    assert captured["data"]["response_format"] == "verbose_json"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert [s.text for s in segs] == ["ok"]


def test_transcribe_file_no_api_key_omits_auth(tmp_path, monkeypatch):
    wav = tmp_path / "chunk.wav"
    wav.write_bytes(b"RIFFfake")
    captured = {}

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"text": "x"}

    def fake_post(url, **kwargs):
        captured["headers"] = kwargs.get("headers")
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = LocalOpenAIProvider("http://localhost:9000", "whisper-1")
    provider.transcribe_file(wav, "vi")
    assert "Authorization" not in captured["headers"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src-python && python -m pytest tests/test_local_openai.py -v`
Expected: FAIL (`_parse_transcription_response` / `transcribe_file` not defined; the stub has no such members).

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `src-python/stt_providers/local_openai.py`:

```python
"""Local, self-hosted OpenAI-compatible transcription provider.

Targets any server exposing POST /v1/audio/transcriptions — faster-whisper-server,
whisper.cpp server, Speaches, or vLLM-served audio models. Whisper large-v3 is the
recommended default for Vietnamese quality. v1 supports batch only."""
from pathlib import Path

import httpx

from .base import STTProvider, STTSegment

# Generous default: a VAD chunk (~22s) on a CPU Whisper server can take a while.
_DEFAULT_TIMEOUT = 300.0


def _parse_transcription_response(payload: dict) -> list[STTSegment]:
    """Map an OpenAI-style transcription response to STTSegment[].

    Prefers verbose_json `segments` (with timestamps). Falls back to a single
    segment from the plain `{"text": ...}` shape when the server ignores
    verbose_json. Returns [] for empty/blank results (never raises on shape)."""
    language = payload.get("language")
    segments = payload.get("segments")
    if isinstance(segments, list) and segments:
        out: list[STTSegment] = []
        for s in segments:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            start = s.get("start")
            end = s.get("end")
            out.append(STTSegment(
                text=text,
                start_ms=int(start * 1000) if start is not None else None,
                end_ms=int(end * 1000) if end is not None else None,
                language=language,
            ))
        if out:
            return out

    text = (payload.get("text") or "").strip()
    if text:
        return [STTSegment(text=text, start_ms=None, end_ms=None, language=language)]
    return []


class LocalOpenAIProvider(STTProvider):
    name = "local"
    supports_batch = True
    supports_realtime = False   # v2
    native_diarization = False  # batch pipeline layers CAM++
    native_translation = False

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout = timeout

    def transcribe_file(self, wav_path: Path | str, language: str, **opts) -> list[STTSegment]:
        url = f"{self.base_url}/v1/audio/transcriptions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = {"model": self.model, "response_format": "verbose_json"}
        if language:
            data["language"] = language
        path = Path(wav_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "audio/wav")}
            resp = httpx.post(url, data=data, files=files, headers=headers,
                              timeout=self.timeout)
        resp.raise_for_status()
        return _parse_transcription_response(resp.json())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src-python && python -m pytest tests/test_local_openai.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/local_openai.py src-python/tests/test_local_openai.py
git commit -m "feat(stt): implement LocalOpenAIProvider HTTP transcription"
```

---

## Task 4: Registry — build local provider + transcriber from settings

This isolates the local-provider wiring so it's testable without importing the
heavy `upload_pipeline` module (which pulls in riva/soniox/onnxruntime).

**Files:**
- Modify: `src-python/stt_providers/registry.py` (create)
- Test: `src-python/tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `src-python/tests/test_registry.py`:

```python
import pytest
from stt_providers import registry
from stt_providers.base import STTSegment


class FakeDB:
    def __init__(self, settings): self._s = settings
    def get_setting(self, key): return self._s.get(key)


def test_build_local_provider_ok():
    db = FakeDB({"local_stt_base_url": "http://x:9000", "local_stt_model": "whisper-1",
                 "local_stt_api_key": "k"})
    p = registry.build_local_provider(db)
    assert p.base_url == "http://x:9000"
    assert p.model == "whisper-1"
    assert p.api_key == "k"


def test_build_local_provider_missing_url():
    db = FakeDB({"local_stt_model": "whisper-1"})
    with pytest.raises(RuntimeError, match="Base URL"):
        registry.build_local_provider(db)


def test_build_local_provider_missing_model():
    db = FakeDB({"local_stt_base_url": "http://x:9000"})
    with pytest.raises(RuntimeError, match="model"):
        registry.build_local_provider(db)


def test_build_local_transcriber_returns_joined_text(monkeypatch):
    db = FakeDB({"local_stt_base_url": "http://x:9000", "local_stt_model": "whisper-1",
                 "stt_language": "vi"})
    monkeypatch.setattr(
        "stt_providers.local_openai.LocalOpenAIProvider.transcribe_file",
        lambda self, path, language, **o: [
            STTSegment(text="xin", start_ms=0, end_ms=1),
            STTSegment(text="chào", start_ms=1, end_ms=2),
        ],
    )
    transcribe = registry.build_local_transcriber(db)
    assert transcribe("/tmp/chunk.wav") == "xin chào"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src-python && python -m pytest tests/test_registry.py -v`
Expected: FAIL (`registry` has no `build_local_provider` / `build_local_transcriber`).

- [ ] **Step 3: Write the implementation**

Create `src-python/stt_providers/registry.py`:

```python
"""Construct STT providers from the SQLite settings (key/value)."""
from typing import Callable

from .base import segments_to_text
from .local_openai import LocalOpenAIProvider


def build_local_provider(db) -> LocalOpenAIProvider:
    """Validate local STT settings and build the provider, or raise a
    user-actionable RuntimeError pointing back to Settings."""
    base_url = (db.get_setting("local_stt_base_url") or "").strip()
    model = (db.get_setting("local_stt_model") or "").strip()
    if not base_url:
        raise RuntimeError(
            "Local STT Base URL chưa được cấu hình. Vào Settings → STT → Local."
        )
    if not model:
        raise RuntimeError(
            "Local STT model chưa được cấu hình. Vào Settings → STT → Local."
        )
    api_key = db.get_setting("local_stt_api_key") or ""
    return LocalOpenAIProvider(base_url, model, api_key)


def build_local_transcriber(db) -> Callable[[str], str]:
    """Return a per-chunk transcribe callable (path -> text) for the batch
    pipeline, matching the (path)->str shape of the Nvidia/Soniox calls."""
    provider = build_local_provider(db)
    language = (db.get_setting("stt_language") or "vi").strip().lower()

    def _transcribe(path: str) -> str:
        return segments_to_text(provider.transcribe_file(path, language))

    return _transcribe
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src-python && python -m pytest tests/test_registry.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite**

Run: `cd src-python && python -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src-python/stt_providers/registry.py src-python/tests/test_registry.py
git commit -m "feat(stt): add registry for local provider + transcriber"
```

---

## Task 5: Wire local provider into the batch pipeline

Refactor the inline STT routing in `_process_chunks_parallel` into a resolver,
then add the `local` branch. This is the only change to working pipeline code.

**Files:**
- Modify: `src-python/services/upload_pipeline.py` (`_process_chunks_parallel`, ~lines 1441-1525)

- [ ] **Step 1: Run impact analysis (repo convention)**

Per `CLAUDE.md`/`gitnexus.md`, before editing run:
```
gitnexus_impact({target: "_process_chunks_parallel", direction: "upstream"})
```
Report the blast radius to the user. Proceed only if not HIGH/CRITICAL without confirmation. (Expected: internal to `upload_pipeline`, called by `_run_nvidia_chunked_pipeline`.)

- [ ] **Step 2: Add the resolver function**

In `src-python/services/upload_pipeline.py`, add this function immediately **above** `async def _process_chunks_parallel(` (around line 1441). It moves the existing nvidia/soniox routing into one place and adds `local`:

```python
def _resolve_chunk_transcriber(db_):
    """Resolve a per-chunk transcribe callable (path -> text) + provider name
    from settings. Centralizes provider routing for the batch chunked path."""
    provider = (db_.get_setting("stt_provider") or "nvidia").strip().lower()
    if provider not in ("nvidia", "soniox", "local"):
        provider = "nvidia"

    if provider == "nvidia":
        nvidia_key = (
            db_.get_setting("nvidia_api_key")
            or os.environ.get("NVIDIA_API_KEY", "")
        )
        if not nvidia_key:
            raise RuntimeError(
                "Nvidia API key chưa được cấu hình. Vào Settings → Nvidia API Key."
            )
        riva_lang = get_language_code(db_.get_setting("stt_language") or "vi")

        def _t(path: str) -> str:
            return transcribe_nvidia_streaming(path, nvidia_key, riva_lang)
        return _t, provider

    if provider == "soniox":
        soniox_key = (
            db_.get_setting("soniox_api_key")
            or os.environ.get("SONIOX_API_KEY", "")
        )
        if not soniox_key:
            raise RuntimeError(
                "Soniox API key chưa được cấu hình. Vào Settings → Soniox API Key."
            )
        hints_raw = db_.get_setting("soniox_language_hints") or "vi"
        soniox_hints = [h.strip() for h in hints_raw.split(",") if h.strip()] or ["vi"]

        def _t(path: str) -> str:
            return transcribe_soniox_file(path, soniox_key, soniox_hints)
        return _t, provider

    # local — OpenAI-compatible transcription endpoint (validation inside)
    from stt_providers.registry import build_local_transcriber
    return build_local_transcriber(db_), provider
```

- [ ] **Step 3: Replace the inline routing block**

In `_process_chunks_parallel`, delete the existing routing block — everything from the comment `# ── STT provider routing ──` down through the line `log.info("[pipeline] STT provider: %s", stt_provider)` (the `if stt_provider == "nvidia": ... else: ...` credential block, ~lines 1460-1520). Replace it with:

```python
    # ── STT provider routing ────────────────────────────────────────────
    # Resolve a per-chunk transcribe callable from the user's Settings choice.
    transcribe_fn, stt_provider = _resolve_chunk_transcriber(db)
    log.info("[pipeline] STT provider: %s", stt_provider)
```

- [ ] **Step 4: Replace the per-chunk STT dispatch**

Inside `_process_one`, replace the `if stt_provider == "nvidia": ... else: ...` block that builds `stt_task` with a single call to the resolved function:

```python
            # Transcribe via the resolved provider callable (nvidia=gRPC,
            # soniox=async file API, local=OpenAI-compatible HTTP).
            stt_task = asyncio.to_thread(transcribe_fn, str(chunk.path))
```

- [ ] **Step 5: Import smoke check**

Run (from `src-python/`, with sidecar deps installed):
```bash
cd src-python && python -c "import services.upload_pipeline; print('import ok')"
```
Expected: `import ok` (no syntax/NameError). If riva/onnxruntime aren't installed in this env, instead verify with: `python -m py_compile services/upload_pipeline.py && echo compile-ok`.

- [ ] **Step 6: Detect changes (repo convention)**

Run `gitnexus_detect_changes()` and confirm only `_process_chunks_parallel` / new `_resolve_chunk_transcriber` are affected, as expected.

- [ ] **Step 7: Commit**

```bash
git add src-python/services/upload_pipeline.py
git commit -m "feat(stt): route batch pipeline through provider resolver + local branch"
```

---

## Task 6: Settings backend — mask key + STT model listing

**Files:**
- Modify: `src-python/api/settings.py`

- [ ] **Step 1: Mask the local STT API key**

In `src-python/api/settings.py`, extend `_SENSITIVE_KEYS`:

```python
_SENSITIVE_KEYS = frozenset({"nvidia_api_key", "llm_api_key", "soniox_api_key", "local_stt_api_key"})
```

- [ ] **Step 2: Add the /stt-models endpoint**

Append to `src-python/api/settings.py` (after the existing `/models` handler):

```python
@router.get("/stt-models")
async def list_stt_models(
    base_url: str = Query(...),
    api_key: str = Query(default=""),
):
    """List models from a local OpenAI-compatible transcription server.
    Best-effort: many Whisper servers don't implement /v1/models — callers
    fall back to free-text model entry on error."""
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return {"models": models}
    except Exception as e:  # noqa: BLE001 — surface any failure as graceful fallback
        log.warning("[settings] /stt-models failed: %s", e)
        return {"error": str(e), "models": []}
```

- [ ] **Step 3: Manual verification**

Start the sidecar (or run app in dev). With any OpenAI-compatible server running at e.g. `http://localhost:9000`, verify:
```bash
curl "http://localhost:8765/stt-models?base_url=http://localhost:9000"
```
Expected: `{"models": [...]}` if the server implements `/v1/models`, else `{"error": "...", "models": []}` (no crash). Also confirm `GET /settings` masks `local_stt_api_key` after saving one.

- [ ] **Step 4: Commit**

```bash
git add src-python/api/settings.py
git commit -m "feat(stt): mask local_stt_api_key + add /stt-models endpoint"
```

---

## Task 7: Frontend — local STT provider tab + config block

**Files:**
- Modify: `src/lib/api.ts`
- Modify: `src/components/SettingsPanel.tsx`

- [ ] **Step 1: Add fetchSttModels to api.ts**

In `src/lib/api.ts`, after the `fetchLLMModels` export, add:

```typescript
export const fetchSttModels = (
    baseUrl: string,
    apiKey?: string,
): Promise<{ models: string[]; error?: string }> => {
    const params = new URLSearchParams({ base_url: baseUrl });
    if (apiKey && !apiKey.includes('•')) params.set('api_key', apiKey);
    return request<{ models: string[]; error?: string }>(`/stt-models?${params}`);
};
```

- [ ] **Step 2: Import it + extend the provider type**

In `src/components/SettingsPanel.tsx`:

Change the import on line 5 to include `fetchSttModels`:
```typescript
import { getSettings, saveSettings, diagnose, fetchLLMModels, fetchSttModels } from '../lib/api';
```

Change line 14:
```typescript
    const [sttProvider, setSttProvider] = useState<'nvidia' | 'soniox' | 'local'>('nvidia');
```

- [ ] **Step 3: Add local state**

After line 18 (`const [nvidiaLang, setNvidiaLang] = useState('vi');`), add:

```typescript
    const [localBaseUrl, setLocalBaseUrl] = useState('');
    const [localModel, setLocalModel] = useState('');
    const [localKey, setLocalKey] = useState('');
    const [showLocalKey, setShowLocalKey] = useState(false);
    const [sttModelOptions, setSttModelOptions] = useState<string[]>([]);
    const [fetchingSttModels, setFetchingSttModels] = useState(false);
```

- [ ] **Step 4: Load local settings**

In `loadSettings`, change the cast on line 94 and add loads after line 101 (`setNvidiaLang(...)`):

```typescript
            setSttProvider((s.stt_provider as 'nvidia' | 'soniox' | 'local') || 'nvidia');
```
```typescript
            if (s.local_stt_base_url) setLocalBaseUrl(s.local_stt_base_url);
            if (s.local_stt_model) setLocalModel(s.local_stt_model);
            if (s.local_stt_api_key) setLocalKey('••••••••');
```

- [ ] **Step 5: Save local settings**

In `buildSettingsBody`, after line 127 (`body.soniox_language_hints = ...`), add:

```typescript
        body.local_stt_base_url = localBaseUrl;
        body.local_stt_model = localModel;
        if (!localKey.includes('•')) body.local_stt_api_key = localKey;
```

- [ ] **Step 6: Add the STT model fetch handler**

After `handleFetchModels` (ends ~line 161), add:

```typescript
    const handleFetchSttModels = async () => {
        if (!localBaseUrl.trim()) return;
        setFetchingSttModels(true);
        setSttModelOptions([]);
        try {
            const result = await fetchSttModels(localBaseUrl, localKey);
            if (result.error) {
                showToast(lang === 'vi' ? `Lỗi lấy model: ${result.error}` : `Fetch error: ${result.error}`, 'error');
            } else if (result.models && result.models.length > 0) {
                setSttModelOptions(result.models);
            } else {
                showToast(lang === 'vi' ? 'Server không trả model — nhập tay tên model' : 'Server returned no models — type the model name', 'warning');
            }
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            showToast(lang === 'vi' ? `Lỗi kết nối: ${msg}` : `Connection error: ${msg}`, 'error');
        }
        setFetchingSttModels(false);
    };
```

- [ ] **Step 7: Add the third provider tab**

In the provider tabs block, after the Soniox `<button>` (closes ~line 351), add a Local tab:

```tsx
                                <button
                                    className={`provider-tab${sttProvider === 'local' ? ' active' : ''}`}
                                    onClick={() => setSttProvider('local')}
                                >
                                    <strong>Local (Whisper)</strong>
                                    <span className="provider-tab-desc">{lang === 'vi' ? 'Tự host, riêng tư' : 'Self-hosted, private'}</span>
                                </button>
```

- [ ] **Step 8: Gate the shared API-Key + Language groups to cloud providers**

The shared "API Key" group (line 367) and "Language Selection" group (line 397) are nvidia/soniox-specific. Wrap each so they only render for cloud providers. Change the opening of the API Key group:

```tsx
                        {/* API Key — Nvidia/Soniox only */}
                        {sttProvider !== 'local' && <div className="setting-group setting-group--full">
```
and its matching closing `</div>` (line 394) to `</div>}`.

Change the opening of the Language group (line 397):
```tsx
                        {/* Language Selection — Nvidia/Soniox dropdown */}
                        {sttProvider !== 'local' && <div className="setting-group">
```
and its matching closing `</div>` (line 431) to `</div>}`.

- [ ] **Step 9: Show Max Speakers for local too (it uses CAM++)**

Change line 434's condition from `sttProvider === 'nvidia'` to include local:

```tsx
                        {sttProvider !== 'soniox' && <div className="setting-group setting-group--full">
```

- [ ] **Step 10: Add the Local config block**

Immediately after the Max Speakers group's closing `</div>}` (line 454) and before the STT section's closing `</div>` (line 455), insert the local block:

```tsx
                        {sttProvider === 'local' && <>
                            <div className="setting-group setting-group--full">
                                <div className="setting-label" style={{ textTransform: 'uppercase' }}>Base URL</div>
                                <input
                                    type="text"
                                    className="setting-input"
                                    value={localBaseUrl}
                                    onChange={(e) => { setLocalBaseUrl(e.target.value); setSttModelOptions([]); }}
                                    placeholder="http://localhost:9000"
                                />
                                <div className="setting-hint">
                                    {lang === 'vi'
                                        ? 'Server tương thích OpenAI (faster-whisper-server, whisper.cpp, Speaches…). Whisper large-v3 cho chất lượng tiếng Việt tốt nhất.'
                                        : 'OpenAI-compatible server (faster-whisper-server, whisper.cpp, Speaches…). Whisper large-v3 gives the best Vietnamese quality.'}
                                </div>
                            </div>

                            <div className="setting-group setting-group--full">
                                <div className="setting-label">
                                    API Key
                                    <ConfigBadge ok={localKey.length > 0} optional />
                                </div>
                                <div className="setting-input-wrap">
                                    <input
                                        type={showLocalKey ? 'text' : 'password'}
                                        className="setting-input"
                                        value={localKey}
                                        onChange={(e) => setLocalKey(e.target.value)}
                                        placeholder={lang === 'vi' ? '(tùy chọn)' : '(optional)'}
                                    />
                                    <button type="button" className="setting-eye-btn" onClick={() => setShowLocalKey(v => !v)} aria-label="toggle key">
                                        {showLocalKey ? (
                                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><line x1="1" x2="23" y1="1" y2="23"/></svg>
                                        ) : (
                                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                                        )}
                                    </button>
                                </div>
                            </div>

                            <div className="setting-group">
                                <div className="setting-label" style={{ textTransform: 'uppercase' }}>
                                    Model <ConfigBadge ok={localModel.length > 0} />
                                </div>
                                <CustomSelect
                                    options={fetchingSttModels
                                        ? [{ value: '', label: lang === 'vi' ? 'Đang tải...' : 'Loading...' }]
                                        : (sttModelOptions.length > 0
                                            ? sttModelOptions.map(m => ({ value: m, label: m }))
                                            : (localModel ? [{ value: localModel, label: localModel }] : [{ value: '', label: lang === 'vi' ? '-- Bấm để tải / nhập tay --' : '-- Click to load / type --' }]))}
                                    value={fetchingSttModels ? '' : localModel}
                                    onChange={setLocalModel}
                                    disabled={fetchingSttModels || localBaseUrl.trim().length === 0}
                                    onOpen={() => {
                                        if (localBaseUrl.trim() && sttModelOptions.length === 0 && !fetchingSttModels) {
                                            handleFetchSttModels();
                                        }
                                    }}
                                />
                                <input
                                    type="text"
                                    className="setting-input"
                                    style={{ marginTop: '6px' }}
                                    value={localModel}
                                    onChange={(e) => setLocalModel(e.target.value)}
                                    placeholder={lang === 'vi' ? 'hoặc nhập tên model (vd: large-v3)' : 'or type model name (e.g. large-v3)'}
                                />
                            </div>

                            <div className="setting-group">
                                <div className="setting-label">{t('primary_language', lang)}</div>
                                <CustomSelect
                                    className="setting-lang-select"
                                    options={sonioxLanguages}
                                    value={nvidiaLang}
                                    onChange={setNvidiaLang}
                                />
                                <div className="setting-hint">{t('language_hint', lang)}</div>
                            </div>
                        </>}
```

> Note: the local block reuses `nvidiaLang`/`setNvidiaLang` (saved as `stt_language`) and the `sonioxLanguages` list (ISO codes like `vi`/`en`/`fr`, exactly what Whisper expects). Model has both a dropdown (auto-fetch on open) and a free-text input as the fallback when the server has no `/v1/models`.

- [ ] **Step 11: Type-check / build**

Run:
```bash
pnpm build
```
Expected: `tsc` passes with no type errors and Vite build succeeds.

- [ ] **Step 12: Commit**

```bash
git add src/lib/api.ts src/components/SettingsPanel.tsx
git commit -m "feat(stt): add local Whisper provider UI in Settings"
```

---

## Task 8: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the local backend**

In `README.md`, in the "🔧 STT Backends" table (both Vietnamese and English sections), add a row for the local provider:

```markdown
| 💻 **Local (Whisper)** | Tự host, riêng tư 100%, không tốn phí API. Chỉ hỗ trợ Upload file (batch) ở phiên bản này | Server tương thích OpenAI (faster-whisper-server / whisper.cpp / Speaches) chạy local | Miễn phí (tự host) |
```

And in the English "STT Backends" section:

```markdown
| 💻 **Local (Whisper)** | Self-hosted, fully private, no API cost. Upload (batch) only in this version | An OpenAI-compatible server (faster-whisper-server / whisper.cpp / Speaches) running locally | Free (self-hosted) |
```

Add a short note under the table:
```markdown
> **Local STT (v1):** point Settings → STT → Local at your OpenAI-compatible transcription server's Base URL. Recommended model: Whisper `large-v3` for Vietnamese. Realtime recording and cabin translation with the local provider are planned for a later version.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document local Whisper STT backend"
```

---

## Task 9: End-to-end manual verification

No automated E2E exists; verify the full path manually.

- [ ] **Step 1: Start a local Whisper server**

Example with faster-whisper-server (or any OpenAI-compatible transcription server) on port 9000. Confirm it serves `POST /v1/audio/transcriptions`.

- [ ] **Step 2: Configure the app**

Run the app (`npx tauri dev`). Settings → STT → **Local (Whisper)** → set Base URL `http://localhost:9000`, Model `large-v3` (or click Model to auto-load), Language `Vietnamese`. Save.

- [ ] **Step 3: Upload a file**

Meetings → Upload file → pick a short vi/en/fr audio file. Verify:
- Pipeline progresses through normalize → split → transcribe → diarize → (summary).
- Transcript appears with per-chunk timestamps and speaker labels (CAM++).
- Vietnamese transcription quality is acceptable.

- [ ] **Step 4: Verify resume**

Start a longer upload, kill the app mid-transcription, reopen the meeting, hit Resume. Confirm only un-transcribed chunks re-POST and the transcript completes.

- [ ] **Step 5: Verify error handling**

Stop the Whisper server, upload a file. Confirm chunks fail gracefully with a clear message and the retry-failed-chunks path works once the server is back.

- [ ] **Step 6: Confirm no regressions**

Switch back to Nvidia, upload a file → still works. Switch to Soniox, upload → still works.

---

## Self-Review (completed during planning)

**Spec coverage:** §3 approach → Tasks 2–5,7. §4.1 package → Tasks 2–4. §4.2 batch integration → Task 5. §4.3 settings/frontend → Tasks 6–7. §5 data flow → Task 5 + Task 9. §6 error handling → Task 5 (reuses per-chunk failure tracking) + Task 9 Step 5. §7 testing → Tasks 2–4 (unit) + Task 9 (manual). §8 limitations → Task 8 docs. §9 files → matches File Structure above.

**Deviations from spec (intentional):** (1) `transcribe_file` is **sync** (wrapped in `asyncio.to_thread`) rather than `async`, to match the existing pipeline's `to_thread` pattern and avoid event-loop nesting. (2) Automated tests cover the `stt_providers` package only (no heavy riva/onnxruntime imports); the `upload_pipeline` wiring is verified by import smoke check + manual E2E — flagged explicitly, not a silent gap.

**Placeholder scan:** none. **Type consistency:** `STTSegment`, `segments_to_text`, `build_local_provider`, `build_local_transcriber`, `transcribe_file`, `_parse_transcription_response`, `fetchSttModels`, `_resolve_chunk_transcriber` used consistently across tasks.
