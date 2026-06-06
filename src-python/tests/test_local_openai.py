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


import httpx
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


def test_parse_malformed_segments_do_not_raise():
    # Non-dict items and non-numeric timestamps must be tolerated, not crash.
    payload = {
        "segments": [
            "not a dict",
            {"start": "0.0", "end": None, "text": "kept"},
        ],
    }
    segs = _parse_transcription_response(payload)
    assert [s.text for s in segs] == ["kept"]
    assert segs[0].start_ms is None and segs[0].end_ms is None


def test_parse_all_blank_segments_falls_back_to_text():
    payload = {"segments": [{"start": 0, "end": 1, "text": "   "}], "text": "fallback"}
    segs = _parse_transcription_response(payload)
    assert [s.text for s in segs] == ["fallback"]


def test_parse_segment_without_timestamps():
    segs = _parse_transcription_response({"segments": [{"text": "hi"}]})
    assert len(segs) == 1
    assert segs[0].start_ms is None and segs[0].end_ms is None


def test_provider_strips_trailing_v1():
    p = LocalOpenAIProvider("http://host:9000/v1", "whisper-1")
    assert p.base_url == "http://host:9000"


def test_provider_strips_trailing_v1_and_slash():
    p = LocalOpenAIProvider("http://host:9000/v1/", "whisper-1")
    assert p.base_url == "http://host:9000"


def test_provider_plain_base_url_unchanged():
    p = LocalOpenAIProvider("http://host:9000", "whisper-1")
    assert p.base_url == "http://host:9000"


def test_provider_supports_realtime_and_opens_session():
    from stt_providers.local_openai import LocalOpenAIProvider
    from stt_providers.local_realtime import LocalRealtimeSession
    p = LocalOpenAIProvider("http://localhost:8000", "whisper-1")
    assert p.supports_realtime is True
    session = p.open_session("en")
    assert isinstance(session, LocalRealtimeSession)
    assert session._language == "en"
