"""On-device batch transcription: whole-file faster-whisper at the best model.

`segments_from_faster_whisper` is the pure mapping (unit-tested).
`transcribe_file_ondevice` is the production entry: load the model (large-v3 by
default) and transcribe the WAV with built-in VAD; the upload pipeline layers
CAM++ diarization on the returned segments. The native model call is E2E-only.
"""
from __future__ import annotations

from pathlib import Path


def segments_from_faster_whisper(segments) -> list[dict]:
    out: list[dict] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        out.append({
            "text": text,
            "start_ms": int((s.start or 0) * 1000),
            "end_ms": int((s.end or 0) * 1000),
        })
    return out


def transcribe_file_ondevice(wav_path, language: str, model_dir,
                             device: str = "cpu", compute_type: str = "int8") -> list[dict]:  # pragma: no cover - native
    from faster_whisper import WhisperModel
    model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(Path(wav_path)), language=language, vad_filter=True, beam_size=5,
    )
    return segments_from_faster_whisper(segments)
