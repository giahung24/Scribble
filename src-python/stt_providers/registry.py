"""Construct STT providers from the SQLite settings (key/value)."""
from typing import Callable

from .base import segments_to_text
from .local_openai import LocalOpenAIProvider


def build_local_provider(db) -> LocalOpenAIProvider:
    """Validate local STT settings and build the provider, or raise a
    user-actionable RuntimeError pointing back to Settings."""
    base_url = (db.get_setting("local_stt_base_url") or "").strip()
    model = (db.get_setting("local_stt_model") or "").strip()
    if not base_url:
        raise RuntimeError(
            "Local STT Base URL chưa được cấu hình. Vào Settings → STT → Local."
        )
    if not model:
        raise RuntimeError(
            "Local STT model chưa được cấu hình. Vào Settings → STT → Local."
        )
    api_key = (db.get_setting("local_stt_api_key") or "").strip()
    return LocalOpenAIProvider(base_url, model, api_key)


def build_local_transcriber(db) -> Callable[[str], str]:
    """Return a per-chunk transcribe callable (path -> text) for the batch
    pipeline, matching the (path)->str shape of the Nvidia/Soniox calls."""
    provider = build_local_provider(db)
    language = (db.get_setting("stt_language") or "vi").strip().lower()

    def _transcribe(path: str) -> str:
        return segments_to_text(provider.transcribe_file(path, language))

    return _transcribe
