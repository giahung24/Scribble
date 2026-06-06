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
