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
