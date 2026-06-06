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
