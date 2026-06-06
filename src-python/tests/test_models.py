from pathlib import Path
from stt_providers.ondevice.models import ModelManager

def test_paths_under_root(tmp_path):
    mm = ModelManager(root=tmp_path)
    assert mm.whisper_dir("small") == tmp_path / "whisper" / "small"
    assert mm.nllb_dir() == tmp_path / "nllb-200-distilled-600M"

def test_is_present_false_then_true(tmp_path):
    mm = ModelManager(root=tmp_path)
    assert mm.is_whisper_present("small") is False
    d = mm.whisper_dir("small"); d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"x")  # non-empty dir with a file = present
    assert mm.is_whisper_present("small") is True

def test_ensure_whisper_downloads_only_when_absent(tmp_path):
    calls = []
    mm = ModelManager(root=tmp_path, downloader=lambda kind, name, dest: calls.append((kind, name, str(dest))) or dest.mkdir(parents=True, exist_ok=True) or (dest / "model.bin").write_bytes(b"x"))
    p1 = mm.ensure_whisper("small")            # absent → downloads
    assert calls == [("whisper", "small", str(mm.whisper_dir("small")))]
    p2 = mm.ensure_whisper("small")            # present → no second download
    assert len(calls) == 1
    assert p1 == p2 == mm.whisper_dir("small")
