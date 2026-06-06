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
    # after endpoint, buffer + policy reset → fresh partial, no stale commit
    committed, partial = asr.feed(np.zeros(16000, dtype=np.int16).tobytes())
    assert committed == []
    assert [w[0] for w in partial] == ["hello", "world"]
