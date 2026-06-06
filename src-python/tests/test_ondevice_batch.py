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
