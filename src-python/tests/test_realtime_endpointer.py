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
