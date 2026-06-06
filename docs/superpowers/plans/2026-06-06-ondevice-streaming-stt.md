# On-device Streaming STT + Translation (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an on-device STT provider (no server) that does both low-latency streaming realtime transcription+translation and best-quality batch/upload transcription, cross-platform, prioritizing Vietnamese.

**Architecture:** One `faster-whisper` (CTranslate2) engine + shared model cache. Realtime = repeated decode of a rolling buffer + a LocalAgreement-2 commit policy → growing partials; translation via CTranslate2/NLLB-200 (re-translated, throttled). Batch = whole-file faster-whisper at large-v3. CAM++ diarization and the existing realtime WS message shapes are reused. Models download on demand.

**Tech Stack:** Python sidecar (faster-whisper, ctranslate2, transformers tokenizer, numpy), FastAPI WebSocket, React/TypeScript frontend (+ frontend-design skill), Rust (Tauri system-audio routing), PyInstaller packaging. Tests: pytest with mocks for logic; manual E2E + packaging smoke for native/model/hardware.

**Spec:** `docs/superpowers/specs/2026-06-06-ondevice-streaming-stt-design.md`

---

## Scope note (read first)

This is a large feature spanning 5 subsystems. Tasks are ordered so the **engine
core (Tasks 0–7) is fully TDD-tested with mocks** and is independently valuable.
Native integration, UI, and packaging (Tasks 8–15) are verified by compile
checks, the frontend-design skill, and **manual E2E** (real models + a rebuild on
each OS) because the heavy native deps, multi-GB models, and GPU/cross-platform
behavior cannot be unit-tested in CI. Each task notes its verification method
honestly.

All commands run from repo root `/home/Ash/projects/Scribble`. Branch:
`feat/ondevice-streaming-stt`. Python tests: `src-python/.venv/bin/python -m
pytest src-python/tests/ -v` (the venv is created in Task 0). Frontend type-check:
`npx tsc --noEmit` (pnpm project — never use npm).

---

## File Structure

**New (Python sidecar):**
- `src-python/stt_providers/__init__.py` — package marker.
- `src-python/stt_providers/ondevice/__init__.py` — subpackage marker.
- `src-python/stt_providers/ondevice/commit_policy.py` — `LocalAgreement` (pure).
- `src-python/stt_providers/ondevice/lang.py` — app-lang ↔ NLLB FLORES code map.
- `src-python/stt_providers/ondevice/translator.py` — `ThrottledRetranslator` (pure) + `NllbTranslator` (CT2).
- `src-python/stt_providers/ondevice/models.py` — `ModelManager` (paths, skip, download-on-demand).
- `src-python/stt_providers/ondevice/streaming_asr.py` — `WhisperStreamingASR` (faster-whisper + LocalAgreement).
- `src-python/stt_providers/ondevice/session.py` — `OnDeviceStreamingSession`.
- `src-python/stt_providers/ondevice/batch.py` — `transcribe_file_ondevice`.

**New (test infra + tests):**
- `src-python/requirements-dev.txt`, `src-python/tests/conftest.py`,
  `src-python/tests/test_*.py`.

**Modified:**
- `src-python/requirements.txt` — add native deps.
- `src-python/main.py` — `/ws/ondevice-stream` handler.
- `src-python/services/upload_pipeline.py` — `ondevice` batch branch.
- `scribble-sidecar.spec` (PyInstaller) — bundle native libs.
- `src/components/SettingsPanel.tsx` — 3-provider selector with differentiation (frontend-design).
- `src/components/recording/recording-constants.ts`, `use-streaming-stt.ts` — routing.
- `src/components/RecordingBar.tsx`, `src/components/UploadAudioModal.tsx` — readiness gates.
- New `src/components/OnDeviceModelManager.tsx` — model download UI.
- `src-tauri/src/lib.rs` — system-audio WS routing for `ondevice`.
- `README.md` — docs.

---

## Task 0: Test infrastructure bootstrap

**Files:**
- Create: `src-python/requirements-dev.txt`, `src-python/tests/conftest.py`, `src-python/tests/__init__.py`

- [ ] **Step 1: Create the dev requirements** — `src-python/requirements-dev.txt`:

```
pytest>=8.0.0
```

- [ ] **Step 2: Create `src-python/tests/conftest.py`** (puts the sidecar root on sys.path so `from stt_providers...` works, exactly like the app imports):

```python
"""Pytest bootstrap: put the sidecar root (src-python/) on sys.path so tests
use the same flat imports the app uses (e.g. `from stt_providers...`)."""
import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))
```

- [ ] **Step 3: Create empty `src-python/tests/__init__.py`** (empty file).

- [ ] **Step 4: Create venv + install dev deps**

Run:
```bash
python3 -m venv src-python/.venv
src-python/.venv/bin/python -m pip install -U pip pytest
```
Expected: pytest installed. (The heavy runtime deps — faster-whisper etc. — are NOT needed for the mocked unit tests; they're installed only for manual E2E.)

- [ ] **Step 5: Create the package markers** — empty files `src-python/stt_providers/__init__.py` and `src-python/stt_providers/ondevice/__init__.py`.

- [ ] **Step 6: Smoke test** — create `src-python/tests/test_smoke.py`:

```python
def test_packages_importable():
    import stt_providers
    import stt_providers.ondevice
    assert stt_providers.ondevice is not None
```

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 7: Commit**

```bash
git add src-python/requirements-dev.txt src-python/tests/ src-python/stt_providers/__init__.py src-python/stt_providers/ondevice/__init__.py
git commit -m "test(ondevice): bootstrap pytest infra + package skeleton"
```

---

## Task 1: LocalAgreement commit policy (pure logic, TDD)

**Files:**
- Create: `src-python/stt_providers/ondevice/commit_policy.py`
- Test: `src-python/tests/test_commit_policy.py`

- [ ] **Step 1: Write the failing test** — `src-python/tests/test_commit_policy.py`:

```python
"""LocalAgreement-2: a word is committed only once two consecutive Whisper
hypotheses agree on it (beyond what's already committed)."""
from stt_providers.ondevice.commit_policy import LocalAgreement

def _w(*words):
    # build (word, start_ms, end_ms) tuples; timings are placeholders here
    return [(w, i * 100, i * 100 + 100) for i, w in enumerate(words)]

def test_nothing_commits_on_first_hypothesis():
    la = LocalAgreement()
    committed = la.add(_w("the", "cat"))
    assert [c[0] for c in committed] == []
    assert [p[0] for p in la.partial()] == ["the", "cat"]

def test_commits_agreed_prefix_of_two_hypotheses():
    la = LocalAgreement()
    la.add(_w("the", "cat"))
    committed = la.add(_w("the", "cat", "sat"))
    assert [c[0] for c in committed] == ["the", "cat"]
    assert [c[0] for c in la.committed] == ["the", "cat"]
    assert [p[0] for p in la.partial()] == ["sat"]

def test_disagreement_stops_commit_at_divergence():
    la = LocalAgreement()
    la.add(_w("i", "scream"))
    committed = la.add(_w("ice", "cream"))   # diverges at word 0
    assert [c[0] for c in committed] == []
    assert [c[0] for c in la.committed] == []

def test_incremental_commit_advances_monotonically():
    la = LocalAgreement()
    la.add(_w("a", "b"))
    la.add(_w("a", "b", "c"))          # commits a, b
    committed = la.add(_w("a", "b", "c", "d"))  # commits c
    assert [c[0] for c in committed] == ["c"]
    assert [c[0] for c in la.committed] == ["a", "b", "c"]

def test_reset_clears_state():
    la = LocalAgreement()
    la.add(_w("a", "b")); la.add(_w("a", "b"))
    la.reset()
    assert la.committed == []
    assert la.partial() == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_commit_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: ... commit_policy`.

- [ ] **Step 3: Implement** — `src-python/stt_providers/ondevice/commit_policy.py`:

```python
"""LocalAgreement-2 commit policy for streaming Whisper.

The streaming ASR re-decodes a growing audio buffer every ~Nms, producing a
fresh hypothesis each time. A word is only trustworthy once it stops changing:
we commit the longest common prefix (by word text) of the two most recent
hypotheses, beyond what is already committed. The uncommitted tail is the live
partial. This is the UFAL whisper_streaming algorithm.

Words are (text, start_ms, end_ms) tuples so callers keep timing for the UI.
"""
from __future__ import annotations

Word = tuple[str, int, int]


class LocalAgreement:
    def __init__(self) -> None:
        self.committed: list[Word] = []
        self._prev: list[Word] = []
        self._latest: list[Word] = []

    def add(self, hypothesis: list[Word]) -> list[Word]:
        """Feed a fresh full hypothesis. Returns the words newly committed."""
        self._latest = list(hypothesis)
        n = len(self.committed)
        prev_tail = self._prev[n:]
        new_tail = hypothesis[n:]
        agreed: list[Word] = []
        for a, b in zip(prev_tail, new_tail):
            if a[0] == b[0]:
                agreed.append(b)
            else:
                break
        self.committed.extend(agreed)
        self._prev = list(hypothesis)
        return agreed

    def partial(self) -> list[Word]:
        """Uncommitted tail of the most recent hypothesis (the live preview)."""
        return self._latest[len(self.committed):]

    def reset(self) -> None:
        self.committed = []
        self._prev = []
        self._latest = []
```

- [ ] **Step 4: Run to verify it passes**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_commit_policy.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/ondevice/commit_policy.py src-python/tests/test_commit_policy.py
git commit -m "feat(ondevice): LocalAgreement-2 streaming commit policy"
```

---

## Task 2: NLLB language mapping + throttled re-translation (pure logic, TDD)

**Files:**
- Create: `src-python/stt_providers/ondevice/lang.py`
- Create: `src-python/stt_providers/ondevice/translator.py` (the `ThrottledRetranslator` part only in this task; `NllbTranslator` lands in Task 4)
- Test: `src-python/tests/test_translator_logic.py`

- [ ] **Step 1: Write the failing test** — `src-python/tests/test_translator_logic.py`:

```python
from stt_providers.ondevice.lang import to_nllb_code
from stt_providers.ondevice.translator import ThrottledRetranslator

def test_nllb_code_mapping():
    assert to_nllb_code("vi") == "vie_Latn"
    assert to_nllb_code("en") == "eng_Latn"
    assert to_nllb_code("fr") == "fra_Latn"
    # unknown falls back to English rather than raising
    assert to_nllb_code("zz") == "eng_Latn"

def test_retranslator_throttles_and_skips_unchanged():
    calls = []
    def fake_translate(text):
        calls.append(text)
        return text.upper()
    # virtual clock so the test is deterministic (no real sleeping)
    now = {"t": 0.0}
    rt = ThrottledRetranslator(fake_translate, interval_s=1.0, clock=lambda: now["t"])

    # first call always runs
    assert rt.maybe("hello") == "HELLO"
    # same text within interval → skipped (returns None = no update)
    assert rt.maybe("hello") is None
    # changed text but still within interval → throttled (skipped)
    now["t"] = 0.5
    assert rt.maybe("hello world") is None
    # interval elapsed → runs
    now["t"] = 1.1
    assert rt.maybe("hello world") == "HELLO WORLD"
    assert calls == ["hello", "hello world"]

def test_retranslator_final_forces_run_even_if_throttled():
    calls = []
    rt = ThrottledRetranslator(lambda t: (calls.append(t) or t[::-1]),
                               interval_s=1.0, clock=lambda: 0.0)
    rt.maybe("abc")            # runs (first)
    # final() ignores throttle + unchanged checks and always translates
    assert rt.final("abcd") == "dcba"
    assert calls == ["abc", "abcd"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_translator_logic.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3a: Implement `src-python/stt_providers/ondevice/lang.py`:**

```python
"""App language code (vi/en/fr/...) → NLLB-200 FLORES-200 code.

Extend this map as more on-device languages are supported. Unknown codes fall
back to English so a missing entry degrades gracefully instead of raising."""
from __future__ import annotations

_NLLB = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn",
}


def to_nllb_code(app_lang: str) -> str:
    return _NLLB.get((app_lang or "").strip().lower()[:2], "eng_Latn")
```

- [ ] **Step 3b: Implement `src-python/stt_providers/ondevice/translator.py`** (this task adds only `ThrottledRetranslator`; `NllbTranslator` is added in Task 4 — leave the file ending ready for that):

```python
"""Incremental translation helpers for on-device realtime.

ThrottledRetranslator wraps any `translate(text)->str` callable with the
re-translation policy used for live captions: translate the growing text at
most once per `interval_s`, skip when the text hasn't changed, and always
translate on `final()` (segment commit). A pluggable `clock` keeps it
deterministic in tests.
"""
from __future__ import annotations

import time as _time
from typing import Callable, Optional


class ThrottledRetranslator:
    def __init__(self, translate: Callable[[str], str], interval_s: float = 0.3,
                 clock: Callable[[], float] = _time.monotonic):
        self._translate = translate
        self._interval = interval_s
        self._clock = clock
        self._last_at = -1e9
        self._last_src = None
        self._last_out = None

    def maybe(self, text: str) -> Optional[str]:
        """Translate `text` if not throttled and changed; else None (no update)."""
        text = (text or "").strip()
        if not text:
            return None
        if text == self._last_src:
            return None
        if (self._clock() - self._last_at) < self._interval and self._last_src is not None:
            return None
        out = self._translate(text)
        self._last_at = self._clock()
        self._last_src = text
        self._last_out = out
        return out

    def final(self, text: str) -> Optional[str]:
        """Force a translation of the committed text, ignoring throttle."""
        text = (text or "").strip()
        if not text:
            return None
        out = self._translate(text)
        self._last_at = self._clock()
        self._last_src = text
        self._last_out = out
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_translator_logic.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/ondevice/lang.py src-python/stt_providers/ondevice/translator.py src-python/tests/test_translator_logic.py
git commit -m "feat(ondevice): NLLB lang map + throttled re-translation policy"
```

---

## Task 3: Model download manager (TDD with mocked download)

**Files:**
- Create: `src-python/stt_providers/ondevice/models.py`
- Test: `src-python/tests/test_models.py`

- [ ] **Step 1: Write the failing test** — `src-python/tests/test_models.py`:

```python
from pathlib import Path
from stt_providers.ondevice.models import ModelManager

def test_paths_under_root(tmp_path):
    mm = ModelManager(root=tmp_path)
    assert mm.whisper_dir("small") == tmp_path / "whisper" / "small"
    assert mm.nllb_dir() == tmp_path / "nllb-200-distilled-600M"

def test_is_present_false_then_true(tmp_path):
    mm = ModelManager(root=tmp_path)
    assert mm.is_whisper_present("small") is False
    d = mm.whisper_dir("small"); d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"x")  # non-empty dir with a file = present
    assert mm.is_whisper_present("small") is True

def test_ensure_whisper_downloads_only_when_absent(tmp_path):
    calls = []
    mm = ModelManager(root=tmp_path, downloader=lambda kind, name, dest: calls.append((kind, name, str(dest))) or dest.mkdir(parents=True, exist_ok=True) or (dest / "model.bin").write_bytes(b"x"))
    p1 = mm.ensure_whisper("small")            # absent → downloads
    assert calls == [("whisper", "small", str(mm.whisper_dir("small")))]
    p2 = mm.ensure_whisper("small")            # present → no second download
    assert len(calls) == 1
    assert p1 == p2 == mm.whisper_dir("small")
```

- [ ] **Step 2: Run to verify it fails**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_models.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** — `src-python/stt_providers/ondevice/models.py`:

```python
"""On-device model cache + download-on-demand manager.

Resolves model directories under a root (default ~/.voicescribe/models) and
fetches models on first use. The actual network fetch is injected as
`downloader(kind, name, dest)` so it is testable; the default uses
huggingface_hub.snapshot_download (added with the runtime deps).

is_*_present treats a non-empty directory as downloaded (good enough; a partial
download is cleaned by callers on failure)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

# faster-whisper accepts these size names directly (it maps to Systran CT2 repos).
WHISPER_SIZES = ("small", "medium", "large-v3")
NLLB_REPO = "facebook/nllb-200-distilled-600M"


def _default_root() -> Path:
    base = os.environ.get("VOICESCRIBE_HOME")
    root = Path(base) if base else (Path.home() / ".voicescribe")
    return root / "models"


def _hf_download(kind: str, name: str, dest: Path) -> None:  # pragma: no cover - network
    from huggingface_hub import snapshot_download
    repo = NLLB_REPO if kind == "nllb" else f"Systran/faster-whisper-{name}"
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(dest))


class ModelManager:
    def __init__(self, root: Optional[Path] = None,
                 downloader: Callable[[str, str, Path], None] = _hf_download):
        self.root = Path(root) if root else _default_root()
        self._download = downloader

    def whisper_dir(self, size: str) -> Path:
        return self.root / "whisper" / size

    def nllb_dir(self) -> Path:
        return self.root / "nllb-200-distilled-600M"

    @staticmethod
    def _present(d: Path) -> bool:
        return d.is_dir() and any(d.iterdir())

    def is_whisper_present(self, size: str) -> bool:
        return self._present(self.whisper_dir(size))

    def is_nllb_present(self) -> bool:
        return self._present(self.nllb_dir())

    def ensure_whisper(self, size: str) -> Path:
        d = self.whisper_dir(size)
        if not self._present(d):
            self._download("whisper", size, d)
        return d

    def ensure_nllb(self) -> Path:
        d = self.nllb_dir()
        if not self._present(d):
            self._download("nllb", "nllb-200-distilled-600M", d)
        return d
```

- [ ] **Step 4: Run to verify it passes**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_models.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/ondevice/models.py src-python/tests/test_models.py
git commit -m "feat(ondevice): model cache + download-on-demand manager"
```

---

## Task 4: NllbTranslator (CTranslate2 integration)

**Files:**
- Modify: `src-python/stt_providers/ondevice/translator.py` (append `NllbTranslator`)
- Test: `src-python/tests/test_nllb_translator.py` (logic via a fake CT2; real CT2 = E2E)

**Verification note:** the CTranslate2 + transformers-tokenizer call sequence is exercised for real only in manual E2E (Task 14). The unit test pins the *token-flow contract* using a fake translator/tokenizer so a future refactor can't silently break argument wiring.

- [ ] **Step 1: Write the failing test** — `src-python/tests/test_nllb_translator.py`:

```python
from stt_providers.ondevice.translator import NllbTranslator

class _FakeTokenizer:
    def __init__(self): self.src_lang = None
    def convert_ids_to_tokens(self, ids): return [f"tok{i}" for i in ids]
    def encode(self, text): return [1, 2, 3]
    def convert_tokens_to_ids(self, toks): return [9, 9]
    def decode(self, ids, skip_special_tokens=True): return "translated"

class _FakeCT2:
    def __init__(self): self.calls = []
    def translate_batch(self, source, target_prefix=None, **kw):
        self.calls.append((source, target_prefix))
        class _R: hypotheses = [["<tgt>", "tok9", "tok9"]]
        return [_R()]

def test_translate_passes_target_prefix_and_decodes():
    tok, ct2 = _FakeTokenizer(), _FakeCT2()
    tr = NllbTranslator(translator=ct2, tokenizer=tok)
    out = tr.translate("xin chào", src="vie_Latn", tgt="eng_Latn")
    assert out == "translated"
    # target language must be passed as the decoder target prefix
    assert ct2.calls[0][1] == [["eng_Latn"]]
    # src_lang set on the tokenizer before encoding
    assert tok.src_lang == "vie_Latn"
```

- [ ] **Step 2: Run to verify it fails**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_nllb_translator.py -v`
Expected: FAIL — `NllbTranslator` not defined.

- [ ] **Step 3: Append `NllbTranslator` to `src-python/stt_providers/ondevice/translator.py`:**

```python
class NllbTranslator:
    """CTranslate2 + NLLB-200 text translator.

    Construct via `NllbTranslator.load(model_dir, device)` for production, or
    pass `translator`/`tokenizer` directly (tests). The NLLB recipe: set the
    tokenizer src_lang, encode → subword tokens, translate_batch with the target
    language as the decoder target_prefix, then decode the output tokens.
    """
    def __init__(self, translator, tokenizer):
        self._ct2 = translator
        self._tok = tokenizer

    @classmethod
    def load(cls, model_dir, device: str = "cpu", compute_type: str = "int8"):  # pragma: no cover - native
        import ctranslate2
        from transformers import AutoTokenizer
        translator = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        return cls(translator, tokenizer)

    def translate(self, text: str, src: str, tgt: str) -> str:
        self._tok.src_lang = src
        tokens = self._tok.convert_ids_to_tokens(self._tok.encode(text))
        results = self._ct2.translate_batch([tokens], target_prefix=[[tgt]])
        out_tokens = results[0].hypotheses[0][1:]  # drop the leading target-lang token
        out_ids = self._tok.convert_tokens_to_ids(out_tokens)
        return self._tok.decode(out_ids, skip_special_tokens=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_nllb_translator.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/ondevice/translator.py src-python/tests/test_nllb_translator.py
git commit -m "feat(ondevice): NllbTranslator (CTranslate2 + NLLB-200)"
```

---

## Task 5: WhisperStreamingASR (faster-whisper + LocalAgreement)

**Files:**
- Create: `src-python/stt_providers/ondevice/streaming_asr.py`
- Test: `src-python/tests/test_streaming_asr.py` (with a fake model; real faster-whisper = E2E)

**Verification note:** the real `faster_whisper.WhisperModel.transcribe(...)` call is exercised only in E2E. The unit test injects a fake transcribe fn to verify the buffer→hypothesis→LocalAgreement wiring and the partial/commit outputs.

- [ ] **Step 1: Write the failing test** — `src-python/tests/test_streaming_asr.py`:

```python
import numpy as np
from stt_providers.ondevice.streaming_asr import WhisperStreamingASR

class _FakeWord:
    def __init__(self, w, s, e): self.word, self.start, self.end = w, s, e

def _segments(*words):
    class _Seg:
        def __init__(self, ws): self.words = [_FakeWord(*w) for w in ws]
    return [_Seg([(w, i*0.1, i*0.1+0.1) for i, w in enumerate(words)])]

def test_decode_emits_partial_then_commits_on_agreement():
    scripted = [
        _segments("the", "cat"),            # decode 1
        _segments("the", "cat", "sat"),     # decode 2 → commit the, cat
    ]
    calls = {"i": 0}
    def fake_transcribe(audio):
        seg = scripted[min(calls["i"], len(scripted)-1)]
        calls["i"] += 1
        return seg
    asr = WhisperStreamingASR(transcribe_fn=fake_transcribe, sample_rate=16000)

    pcm = np.zeros(16000, dtype=np.int16).tobytes()  # 1s of audio per feed
    committed1, partial1 = asr.feed(pcm)
    assert [c[0] for c in committed1] == []
    assert [p[0] for p in partial1] == ["the", "cat"]

    committed2, partial2 = asr.feed(pcm)
    assert [c[0] for c in committed2] == ["the", "cat"]
    assert [p[0] for p in partial2] == ["sat"]

def test_endpoint_commits_remaining_and_resets():
    def fake_transcribe(audio):
        return _segments("hello", "world")
    asr = WhisperStreamingASR(transcribe_fn=fake_transcribe, sample_rate=16000)
    asr.feed(np.zeros(16000, dtype=np.int16).tobytes())
    asr.feed(np.zeros(16000, dtype=np.int16).tobytes())  # agreement commits both
    final_words = asr.endpoint()
    assert [w[0] for w in final_words] == ["hello", "world"]
    # after endpoint, buffer + policy reset → fresh partial
    _, partial = asr.feed(np.zeros(16000, dtype=np.int16).tobytes())
    assert isinstance(partial, list)
```

- [ ] **Step 2: Run to verify it fails**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_streaming_asr.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** — `src-python/stt_providers/ondevice/streaming_asr.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_streaming_asr.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/ondevice/streaming_asr.py src-python/tests/test_streaming_asr.py
git commit -m "feat(ondevice): streaming Whisper ASR with LocalAgreement"
```

---

## Task 6: OnDeviceStreamingSession (TDD with mocked ASR + translator)

**Files:**
- Create: `src-python/stt_providers/ondevice/session.py`
- Test: `src-python/tests/test_ondevice_session.py`

This mirrors the session lifecycle pattern proven for the other realtime
streamers: `start/feed_audio/results/stop`, bounded worker, sentinel-terminated
`results()`. It detects endpoints with a small energy-based silence detector
(reuse the same RMS idea as the rest of the realtime code: ~600 ms trailing
silence closes a segment).

- [ ] **Step 1: Write the failing test** — `src-python/tests/test_ondevice_session.py`:

```python
import array
from stt_providers.ondevice.session import OnDeviceStreamingSession

SR = 16000
def _speech(ms): return array.array("h", [8000]*int(SR*ms/1000)).tobytes()
def _silence(ms): return array.array("h", [0]*int(SR*ms/1000)).tobytes()

class _FakeASR:
    """Commits the whole hypothesis on the 2nd feed; endpoint returns committed."""
    def __init__(self):
        self._fed = 0
        self.committed = []
    def feed(self, pcm):
        self._fed += 1
        if self._fed >= 2:
            words = [("hello", 0, 100), ("world", 100, 200)]
            self.committed = words
            return words, []
        return [], [("hello", 0, 100)]
    def endpoint(self):
        w = self.committed; self.committed = []; return w

def test_session_streams_partials_then_final_and_terminates():
    asr = _FakeASR()
    session = OnDeviceStreamingSession(asr=asr, translate=None, sample_rate=SR)
    session.start()
    session.feed_audio(_speech(1000))                 # partial
    session.feed_audio(_speech(1000))                 # commit
    session.feed_audio(_silence(700))                 # endpoint → final
    session.stop()
    results = list(session.results())
    finals = [r for r in results if r.get("is_final")]
    assert any("hello world" in (r.get("text") or "") for r in finals)

def test_session_stop_without_audio_terminates():
    session = OnDeviceStreamingSession(asr=_FakeASR(), translate=None, sample_rate=SR)
    session.start(); session.stop()
    assert list(session.results()) == [] or list(session.results()) is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_ondevice_session.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** — `src-python/stt_providers/ondevice/session.py`:

```python
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
        while True:
            item = self._in_q.get()
            if item is _SENTINEL:
                final = self._asr.endpoint()
                self._emit_final(final, bytes(seg_pcm))
                break
            seg_pcm.extend(item)
            committed, partial = self._asr.feed(item)
            if partial or committed:
                preview = " ".join(w[0] for w in (list(committed) + list(partial))).strip()
                if preview:
                    self._out_q.put({"text": preview, "is_final": False})
            # endpoint detection on trailing silence
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
                final = self._asr.endpoint()
                self._emit_final(final, bytes(seg_pcm))
                seg_pcm = bytearray()
                trailing_silence = 0
        self._out_q.put(None)
```

- [ ] **Step 4: Run to verify it passes**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_ondevice_session.py -v`
Expected: PASS (2 passed). If a test hangs, the worker/sentinel wiring has a bug — fix it; do not add sleeps.

- [ ] **Step 5: Commit**

```bash
git add src-python/stt_providers/ondevice/session.py src-python/tests/test_ondevice_session.py
git commit -m "feat(ondevice): realtime streaming session (worker + endpointing)"
```

---

## Task 7: Batch transcriber (TDD with mocked faster-whisper)

**Files:**
- Create: `src-python/stt_providers/ondevice/batch.py`
- Test: `src-python/tests/test_ondevice_batch.py`

- [ ] **Step 1: Write the failing test** — `src-python/tests/test_ondevice_batch.py`:

```python
from stt_providers.ondevice.batch import segments_from_faster_whisper

class _Seg:
    def __init__(self, text, start, end): self.text, self.start, self.end = text, start, end

def test_maps_segments_to_dicts():
    fake = [_Seg(" Hello ", 0.0, 1.2), _Seg("world", 1.2, 2.0), _Seg("   ", 2.0, 2.1)]
    out = segments_from_faster_whisper(fake)
    assert out == [
        {"text": "Hello", "start_ms": 0, "end_ms": 1200},
        {"text": "world", "start_ms": 1200, "end_ms": 2000},
    ]  # blank-only segment dropped, text stripped, seconds→ms
```

- [ ] **Step 2: Run to verify it fails**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_ondevice_batch.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** — `src-python/stt_providers/ondevice/batch.py`:

```python
"""On-device batch transcription: whole-file faster-whisper at the best model.

`segments_from_faster_whisper` is the pure mapping (unit-tested).
`transcribe_file_ondevice` is the production entry: load the model (large-v3 by
default) and transcribe the WAV with built-in VAD; the upload pipeline layers
CAM++ diarization on the returned segments. The native model call is E2E-only.
"""
from __future__ import annotations

from pathlib import Path


def segments_from_faster_whisper(segments) -> list[dict]:
    out: list[dict] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        out.append({
            "text": text,
            "start_ms": int((s.start or 0) * 1000),
            "end_ms": int((s.end or 0) * 1000),
        })
    return out


def transcribe_file_ondevice(wav_path, language: str, model_dir,
                             device: str = "cpu", compute_type: str = "int8") -> list[dict]:  # pragma: no cover - native
    from faster_whisper import WhisperModel
    model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(Path(wav_path)), language=language, vad_filter=True, beam_size=5,
    )
    return segments_from_faster_whisper(segments)
```

- [ ] **Step 4: Run to verify it passes**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_ondevice_batch.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src-python/stt_providers/ondevice/batch.py src-python/tests/test_ondevice_batch.py
git commit -m "feat(ondevice): batch whole-file transcription mapping"
```

---

## Task 8: Runtime dependencies + hardware/model selection helper

**Files:**
- Modify: `src-python/requirements.txt`
- Create: `src-python/stt_providers/ondevice/runtime.py`
- Test: `src-python/tests/test_runtime.py`

- [ ] **Step 1: Add deps** — append to `src-python/requirements.txt`:

```
# On-device STT (faster-whisper realtime + batch) and translation (NLLB via CTranslate2)
faster-whisper>=1.0.0
ctranslate2>=4.0.0
transformers>=4.40.0
huggingface_hub>=0.23.0
sentencepiece>=0.2.0
```

- [ ] **Step 2: Write the failing test** — `src-python/tests/test_runtime.py`:

```python
from stt_providers.ondevice.runtime import pick_device, realtime_model_size

def test_pick_device_prefers_cuda_when_available():
    assert pick_device(cuda_available=lambda: True) == ("cuda", "float16")
    assert pick_device(cuda_available=lambda: False) == ("cpu", "int8")

def test_realtime_model_size_by_device():
    assert realtime_model_size("cuda") == "large-v3"
    assert realtime_model_size("cpu") == "small"   # CPU default (configurable)

def test_realtime_model_size_respects_override():
    assert realtime_model_size("cpu", override="medium") == "medium"
```

- [ ] **Step 3: Run to verify it fails**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/test_runtime.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement** — `src-python/stt_providers/ondevice/runtime.py`:

```python
"""Hardware detection + model-size selection for on-device realtime.

pick_device probes CUDA (injected for tests) and returns (device, compute_type).
realtime_model_size chooses the latency-tuned default per device; batch always
uses large-v3 (decided in the spec) and does not go through this function.
"""
from __future__ import annotations

from typing import Callable


def _cuda_available() -> bool:  # pragma: no cover - native
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def pick_device(cuda_available: Callable[[], bool] = _cuda_available):
    if cuda_available():
        return ("cuda", "float16")
    return ("cpu", "int8")


def realtime_model_size(device: str, override: str | None = None) -> str:
    if override:
        return override
    return "large-v3" if device == "cuda" else "small"
```

- [ ] **Step 5: Run to verify it passes + full suite**

Run: `src-python/.venv/bin/python -m pytest src-python/tests/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src-python/requirements.txt src-python/stt_providers/ondevice/runtime.py src-python/tests/test_runtime.py
git commit -m "feat(ondevice): runtime device/model selection + deps"
```

---

## Task 9: `/ws/ondevice-stream` WebSocket handler

**Files:**
- Modify: `src-python/main.py`

**Verification:** `py_compile` + manual E2E (no unit test — same as the existing
nvidia/soniox handlers). Mirror the existing `nvidia_stream_ws` handler.

- [ ] **Step 1: Study the template** — read `nvidia_stream_ws` in `src-python/main.py` (find it: `grep -n 'async def nvidia_stream_ws' src-python/main.py`). The new handler copies its structure: accept; reset `diarizer`; optional `.pcm` archive from `meeting_id`; an outer-scope `translate_state`; a daemon `_read_results` thread bridging `session.results()` → asyncio queue; an async `_send_results` that sends partial/final/translation/terminal-error JSON and diarizes finals from `result["pcm"]` via `diarizer.identify_speaker_from_samples`; a receive loop feeding bytes to `session.feed_audio` + archive + handling `STOP`/`TRANSLATE:`; a finally that calls `session.stop()`, joins, closes.

- [ ] **Step 2: Add the handler.** Insert as a new module-level function after `nvidia_stream_ws` (column 0, not nested). Build the session like this (the parts that differ from nvidia):

```python
# ─── On-device (faster-whisper) Streaming WebSocket ───
@app.websocket("/ws/ondevice-stream")
async def ondevice_stream_ws(websocket: WebSocket):
    """On-device realtime STT (+ translation) via faster-whisper streaming.
    Mirrors nvidia_stream_ws delivery shapes; ASR + MT run fully locally."""
    await websocket.accept()
    diarizer.reset()
    diarizer.set_source(websocket.query_params.get("source", "web"))
    _max = db.get_setting("max_speakers")
    if _max:
        try:
            diarizer.set_max_speakers(int(_max))
        except (ValueError, TypeError):
            pass

    stt_lang = db.get_setting("stt_language") or "vi"
    translate_state = {"lang": websocket.query_params.get("translate_lang", "")}

    # Build engine. Missing models / load failure → terminal error + close.
    try:
        from stt_providers.ondevice.runtime import pick_device, realtime_model_size
        from stt_providers.ondevice.models import ModelManager
        from stt_providers.ondevice.streaming_asr import WhisperStreamingASR
        from stt_providers.ondevice.translator import NllbTranslator
        from stt_providers.ondevice.lang import to_nllb_code
        from stt_providers.ondevice.session import OnDeviceStreamingSession

        device, compute = pick_device()
        size = realtime_model_size(device, override=db.get_setting("ondevice_model_size") or None)
        mm = ModelManager()
        if not mm.is_whisper_present(size):
            raise RuntimeError(f"On-device model '{size}' chưa tải. Vào Settings → On-device để tải.")
        asr = WhisperStreamingASR.load(mm.whisper_dir(size), language=stt_lang,
                                       device=device, compute_type=compute)
        translate_fn = None
        if translate_state["lang"]:
            if not mm.is_nllb_present():
                raise RuntimeError("Model dịch (NLLB) chưa tải. Vào Settings → On-device để tải.")
            nllb = NllbTranslator.load(mm.nllb_dir(), device=device, compute_type=compute)
            src, tgt = to_nllb_code(stt_lang), to_nllb_code(translate_state["lang"])
            translate_fn = lambda t, _s=src, _t=tgt: nllb.translate(t, _s, _t)
        session = OnDeviceStreamingSession(asr=asr, translate=translate_fn)
    except Exception as e:
        await websocket.send_json({"error": True, "terminal": True, "text": str(e),
                                   "is_final": True, "speaker": "System", "speaker_id": -1})
        await websocket.close()
        return
```

Then reuse the nvidia handler's `_read_results` thread, `_send_results` loop
(partial → send `{is_final:False}`; final → diarize `result["pcm"]`, send, send
`{type:"translation", ...}` when `result.get("translation")`), receive loop
(`session.feed_audio`, STOP, TRANSLATE), and finally block (`session.stop()`,
join, close), **identical in shape** to nvidia. Use `chunk_id` per final like
nvidia.

- [ ] **Step 3: Verify compile + route + suite**

```bash
src-python/.venv/bin/python -m py_compile src-python/main.py
grep -n 'ws/ondevice-stream' src-python/main.py     # exactly one decorator match, column 0
src-python/.venv/bin/python -m pytest src-python/tests/ -q
```
Expected: compile exit 0; one match; suite green.

- [ ] **Step 4: Commit**

```bash
git add src-python/main.py
git commit -m "feat(ondevice): /ws/ondevice-stream realtime handler"
```

---

## Task 10: Upload pipeline `ondevice` batch branch

**Files:**
- Modify: `src-python/services/upload_pipeline.py`

**Verification:** `py_compile` + manual E2E. On `main`, the provider dispatch is
at `src-python/services/upload_pipeline.py:457` (`if stt_provider == "soniox"
... else <nvidia chunked>`).

- [ ] **Step 1: Add an `ondevice` branch.** At the dispatch (around line 457), change the `if/else` to route `ondevice` to a new pipeline function:

```python
    stt_provider = (db.get_setting("stt_provider") or "nvidia").strip().lower()
    if stt_provider == "soniox":
        transcript_parts = await _run_soniox_pipeline(job, meeting_id, wav_path, duration_sec, tmp_root)
        # ... existing soniox post-handling unchanged ...
    elif stt_provider == "ondevice":
        transcript_parts = await _run_ondevice_pipeline(job, meeting, meeting_id, wav_path, tmp_root)
    else:
        transcript_parts = await _run_nvidia_chunked_pipeline(job, meeting, meeting_id, wav_path, tmp_root)
```

- [ ] **Step 2: Add `_run_ondevice_pipeline`** near `_run_nvidia_chunked_pipeline`. It transcribes the whole normalized WAV at large-v3, then layers CAM++ diarization per segment (reuse the same diarization the nvidia path uses on segment audio), producing the same `transcript_parts` shape:

```python
async def _run_ondevice_pipeline(job, meeting, meeting_id, wav_path, tmp_root):
    """On-device batch: whole-file faster-whisper (large-v3) + CAM++ diarization.
    No server, best quality (latency irrelevant for a file)."""
    from stt_providers.ondevice.runtime import pick_device
    from stt_providers.ondevice.models import ModelManager
    from stt_providers.ondevice.batch import transcribe_file_ondevice

    device, compute = pick_device()
    mm = ModelManager()
    if not mm.is_whisper_present("large-v3"):
        # Batch insists on best quality; require the model (UI prompts download).
        raise RuntimeError("On-device model 'large-v3' chưa tải. Vào Settings → On-device để tải.")
    stt_lang = (db.get_setting("stt_language") or "vi")
    await registry.update(job.job_id, progress=40, message="Nhận dạng (on-device)")
    segs = await asyncio.to_thread(
        transcribe_file_ondevice, str(wav_path), stt_lang, str(mm.whisper_dir("large-v3")),
        device, compute,
    )
    # Layer CAM++ diarization per segment + assemble transcript_parts in the
    # same shape the nvidia path persists (reuse the existing diarization helper
    # used by _run_nvidia_chunked_pipeline on segment PCM).
    return _segments_to_transcript_parts_with_diarization(segs, wav_path)
```

  **Implementation note for the engineer:** `_segments_to_transcript_parts_with_diarization` should reuse whatever diarization assembly `_run_nvidia_chunked_pipeline` already uses (read that function and factor out / call the same speaker-assignment + parts-building logic; do not duplicate the CAM++ logic). If extracting a shared helper is too invasive, mirror the parts-building shape (`{text, speaker, speakerId, chunkId}`) and call `diarizer.identify_speaker_from_samples` on each segment's audio sliced from `wav_path`.

- [ ] **Step 3: Allow `ondevice` in the chunk-resolver guard** (around line 1466, `if stt_provider not in ("nvidia", "soniox")`): leave it as-is — `ondevice` does not use the chunk resolver (it has its own pipeline function), so no change needed there. Confirm by reading: the `ondevice` branch returns before reaching that code.

- [ ] **Step 4: Verify compile + suite**

```bash
src-python/.venv/bin/python -m py_compile src-python/services/upload_pipeline.py
src-python/.venv/bin/python -m pytest src-python/tests/ -q
```
Expected: compile clean; suite green.

- [ ] **Step 5: Commit**

```bash
git add src-python/services/upload_pipeline.py
git commit -m "feat(ondevice): on-device batch upload pipeline branch"
```

---

## Task 11: Frontend — provider selector redesign (USE frontend-design skill)

**Files:**
- Modify: `src/components/SettingsPanel.tsx`

**REQUIRED SUB-SKILL:** the implementer subagent MUST invoke the `frontend-design`
skill for this task. The goal is a polished, distinctive provider selector that
**makes the three providers' differences legible at a glance**, cohesive with the
app's existing Tailwind design language (do NOT restyle unrelated UI).

- [ ] **Step 1: Read current provider UI** — read `src/components/SettingsPanel.tsx`'s STT provider section (it currently offers `nvidia` and `soniox`). Note the existing tab/card styling, the `sttProvider` state, and `buildSettingsBody()`.

- [ ] **Step 2: Invoke `frontend-design`** with this content brief (the differentiation copy to communicate — exact words can be refined for tone, but the *distinctions* must come through):

  | Provider | Tagline | Cost | Latency / mode | Privacy | Needs |
  |---|---|---|---|---|---|
  | **Nvidia Riva** | Free, low-latency cloud | Free | Realtime, token-by-token | Cloud | Nvidia API key |
  | **Soniox** | Premium cloud accuracy | Paid | Realtime + native translation | Cloud | Soniox API key |
  | **On-device** | Private, no server, runs locally | Free | Realtime (≈1–2s) **and** Upload (best quality) | 100% local | One-time model download |

  Design requirements:
  - Three selectable cards/tabs (extend the existing two). Selecting `ondevice`
    sets `sttProvider = 'ondevice'` and persists via `buildSettingsBody()` →
    `body.stt_provider`.
  - Each card surfaces its **key differentiator** (cloud-vs-local, free-vs-paid,
    needs-key-vs-needs-download) so a user instantly understands the tradeoff.
  - On-device card, when selected, reveals: language (reuse existing), realtime
    model size (`small`/`medium`/`large-v3`, persisted as `ondevice_model_size`)
    with a note that large-v3 needs a GPU for realtime, and a slot for the model
    manager component (Task 12). No API-key/Base-URL fields.
  - Stay within the app's existing visual system; apply frontend-design polish to
    the comparison, not a wholesale theme change.

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: `TypeScript: No errors found`.

- [ ] **Step 4: Commit**

```bash
git add src/components/SettingsPanel.tsx
git commit -m "feat(ondevice): provider selector with Nvidia/Soniox/On-device differentiation"
```

---

## Task 12: Frontend — model manager UI + routing + readiness gates

**Files:**
- Create: `src/components/OnDeviceModelManager.tsx`
- Modify: `src/components/recording/recording-constants.ts`, `src/components/recording/use-streaming-stt.ts`, `src/components/RecordingBar.tsx`, `src/components/UploadAudioModal.tsx`
- (Backend) Modify: `src-python/main.py` — add `GET /ondevice/models` (status) + `POST /ondevice/models/download` (trigger) endpoints.

- [ ] **Step 1: Backend status/download endpoints** in `src-python/main.py` (small REST handlers using `ModelManager`): `GET /ondevice/models` returns `{whisper: {small:bool, medium:bool, "large-v3":bool}, nllb:bool}` via `is_*_present`; `POST /ondevice/models/download` body `{kind, size?}` runs `ensure_whisper`/`ensure_nllb` in a thread and streams/returns progress (simplest: return 200 when done; the UI polls `GET`). Verify with `py_compile`.

- [ ] **Step 2: WS routing** — `recording-constants.ts`: add `export const WS_PATH_ONDEVICE = "/ws/ondevice-stream";`. In `use-streaming-stt.ts`, import it and route:

```typescript
    const wsPath =
        provider === "soniox" ? WS_PATH_SONIOX
        : provider === "ondevice" ? WS_PATH_ONDEVICE
        : WS_PATH_NVIDIA;
```

- [ ] **Step 3: Model manager component** — `OnDeviceModelManager.tsx`: fetches `GET /ondevice/models`, lists the required models for the current language (Whisper size + NLLB if translation on), shows downloaded/missing, and a Download button that calls `POST /ondevice/models/download` then re-polls. Keep it cohesive with existing settings UI (this is utility UI; frontend-design polish optional here).

- [ ] **Step 4: Readiness gates.** In `RecordingBar.tsx`'s pre-record validation (the block that checks `provider === 'nvidia'`/`'soniox'`), add:

```typescript
                if (provider === 'ondevice') {
                    const r = await fetch('http://127.0.0.1:8765/ondevice/models').then(x => x.json()).catch(() => null);
                    const size = (settings.ondevice_model_size || 'small');
                    const ok = r && r.whisper && r.whisper[size];
                    if (!ok) {
                        showToast(lang === 'vi'
                            ? 'Model on-device chưa tải. Vào Cài đặt → On-device để tải trước khi ghi âm.'
                            : 'On-device model not downloaded. Open Settings → On-device to download first.', 'error');
                        useAppStore.getState().setSettingsOpen(true);
                        return;
                    }
                }
```

  In `UploadAudioModal.tsx`'s `checkSttProviderConfigured`, add an `ondevice`
  branch that checks `large-v3` is present (batch uses large-v3) via the same
  endpoint, returning a clear "download the large-v3 model" message if missing.

- [ ] **Step 5: Type-check + commit**

```bash
npx tsc --noEmit
git add src/components/OnDeviceModelManager.tsx src/components/recording/recording-constants.ts src/components/recording/use-streaming-stt.ts src/components/RecordingBar.tsx src/components/UploadAudioModal.tsx src-python/main.py
git commit -m "feat(ondevice): model manager UI, WS routing, readiness gates"
```

---

## Task 13: Rust — system-audio routing for `ondevice`

**Files:**
- Modify: `src-tauri/src/lib.rs`

**Verification:** inspection (no `cargo` in CI sandbox) + the OS rebuild in Task 15.

- [ ] **Step 1: Route the provider.** In `system_audio_ws_loop`, the WS path is chosen by `stt_provider`. Replace the two-branch selection with a three-way one so `ondevice` system audio hits the right endpoint:

```rust
    let ws_path = match stt_provider {
        "soniox" => "/ws/soniox-stream",
        "ondevice" => "/ws/ondevice-stream",
        _ => "/ws/nvidia-stream",
    };
```

  (Find the current line: `grep -n '/ws/soniox-stream' src-tauri/src/lib.rs`.)

- [ ] **Step 2: Verify + commit**

```bash
grep -n 'ondevice-stream' src-tauri/src/lib.rs     # confirm the new arm
git add src-tauri/src/lib.rs
git commit -m "feat(ondevice): route system-audio capture to /ws/ondevice-stream"
```

---

## Task 14: Manual E2E verification (real models + hardware)

**No code.** This task is the honest verification of everything the unit tests
mocked. The implementer reports results; failures become follow-up fixes.

- [ ] Install runtime deps into the venv: `src-python/.venv/bin/python -m pip install -r src-python/requirements.txt`.
- [ ] Download models via the Settings UI (or `ModelManager.ensure_*`): Whisper `small` + `large-v3` + NLLB.
- [ ] **Realtime:** select On-device, record VI/EN/FR with pauses. Confirm growing partials appear (~1–2s), segments commit, speaker labels populate, and (translation on) translated text updates near-instantly. Confirm STOP/disconnect ends cleanly.
- [ ] **Batch:** upload a VI/EN/FR file with On-device selected, no server running. Confirm large-v3 transcript + diarization, correct output.
- [ ] **Readiness:** with models absent, recording and upload show the download prompt.
- [ ] **GPU (if available):** confirm CUDA is auto-selected (large-v3 realtime) and latency improves.
- [ ] Report latency, VI/EN/FR quality, and any errors.

---

## Task 15: Cross-platform packaging (PyInstaller) + smoke

**Files:**
- Modify: `scribble-sidecar.spec`

**Verification:** a sidecar build + import-and-decode smoke on each OS (the
highest-risk item). This is done on real build machines / CI, not in this
sandbox.

- [ ] **Step 1: Bundle native libs.** In `scribble-sidecar.spec`, ensure `faster_whisper`, `ctranslate2`, `transformers`, `tokenizers`, `sentencepiece`, and `huggingface_hub` are collected. Add to the spec's `Analysis` via `collect_all`:

```python
from PyInstaller.utils.hooks import collect_all
for _pkg in ("faster_whisper", "ctranslate2", "transformers", "tokenizers", "sentencepiece", "huggingface_hub"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d; binaries += _b; hiddenimports += _h
```

  (Match the spec's existing variable names; read the current `scribble-sidecar.spec` first. `transformers` is large — if bundle size is unacceptable, the engineer may switch the NLLB tokenizer to a `sentencepiece`-only path and drop `transformers`; note this in the commit.)

- [ ] **Step 2: Build smoke (per OS).** Build the sidecar and run a smoke that imports the engine and decodes a 1s silent WAV with a tiny model, confirming native libs load. Record pass/fail per OS (Windows, Linux, macOS).

- [ ] **Step 3: Commit**

```bash
git add scribble-sidecar.spec
git commit -m "build(ondevice): bundle faster-whisper/ctranslate2/nllb native libs"
```

---

## Task 16: Docs

**Files:**
- Modify: `README.md`

- [ ] Add an "On-device (Realtime + Upload)" STT section: 100% local, no server; realtime streaming (~1–2s on CPU, token-by-token on GPU) + best-quality upload (large-v3); one-time model download; VI/EN/FR; GPU auto-upgrade; macOS/Win/Linux. Note Phase 2 (SeamlessStreaming GPU, macOS Metal). Commit `docs: on-device STT (realtime + upload)`.

---

## Notes for the executor

- **Order matters:** Tasks 0–8 are pure/mocked TDD and must be green before the
  integration tasks. Tasks 9–13 are integration (compile + E2E). Task 14 is the
  real verification. Task 15 (packaging) is the riskiest and may surface
  bundling fixes.
- **Heavy deps are NOT required for Tasks 0–8** — keep them out of the unit-test
  venv so the logic suite stays fast. Install them only for Task 14.
- **Re-index GitNexus** (`npx gitnexus analyze`) is optional and only for impact
  analysis; it is not required to run the plan.
