"""Local, self-hosted OpenAI-compatible transcription provider.

Targets any server exposing POST /v1/audio/transcriptions — faster-whisper-server,
whisper.cpp server, Speaches, or vLLM-served audio models. Whisper large-v3 is the
recommended default for Vietnamese quality. v1 supports batch only."""
from pathlib import Path

import httpx

from .base import STTProvider, STTSegment

# Generous default: a VAD chunk (~22s) on a CPU Whisper server can take a while.
_DEFAULT_TIMEOUT = 300.0


def normalize_base_url(url: str) -> str:
    """Normalize a user-entered base URL to the server root.

    Strips trailing slashes and a single trailing '/v1' segment, so that both
    'http://host:9000' and 'http://host:9000/v1' work — the app's LLM config
    convention includes '/v1', so users often paste it here too. Callers then
    append '/v1/audio/transcriptions' or '/v1/models' themselves."""
    u = (url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")]
    return u


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
            if not isinstance(s, dict):
                continue
            text = (s.get("text") or "").strip()
            if not text:
                continue
            start = s.get("start")
            end = s.get("end")
            out.append(STTSegment(
                text=text,
                start_ms=int(start * 1000) if isinstance(start, (int, float)) else None,
                end_ms=int(end * 1000) if isinstance(end, (int, float)) else None,
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
    supports_realtime = True    # v2: pseudo-realtime via VAD-chunked batch
    native_diarization = False  # batch pipeline layers CAM++
    native_translation = False

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = _DEFAULT_TIMEOUT):
        self.base_url = normalize_base_url(base_url)
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

    def open_session(self, language: str, **opts):
        """Realtime session — buffers live PCM and transcribes utterances
        through this same provider's batch endpoint. See LocalRealtimeSession."""
        from .local_realtime import LocalRealtimeSession
        return LocalRealtimeSession(self, language)
