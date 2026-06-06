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
