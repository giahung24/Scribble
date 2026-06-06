"""App language code (vi/en/fr/...) → NLLB-200 FLORES-200 code.

Extend this map as more on-device languages are supported. Unknown codes fall
back to English so a missing entry degrades gracefully instead of raising."""
from __future__ import annotations

_NLLB = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn",
}


def to_nllb_code(app_lang: str) -> str:
    return _NLLB.get((app_lang or "").strip().lower()[:2], "eng_Latn")
