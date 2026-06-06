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


def test_build_local_provider_strips_trailing_slash():
    db = FakeDB({"local_stt_base_url": "http://x:9000/", "local_stt_model": "whisper-1"})
    p = registry.build_local_provider(db)
    assert p.base_url == "http://x:9000"
