"""Incremental translation helpers for on-device realtime.

ThrottledRetranslator wraps any `translate(text)->str` callable with the
re-translation policy used for live captions: translate the growing text at
most once per `interval_s`, skip when the text hasn't changed, and always
translate on `final()` (segment commit). A pluggable `clock` keeps it
deterministic in tests.
"""
from __future__ import annotations

import time as _time
from typing import Callable, Optional


class ThrottledRetranslator:
    def __init__(self, translate: Callable[[str], str], interval_s: float = 0.3,
                 clock: Callable[[], float] = _time.monotonic):
        self._translate = translate
        self._interval = interval_s
        self._clock = clock
        self._last_at = -1e9
        self._last_src = None
        self._last_out = None

    def maybe(self, text: str) -> Optional[str]:
        """Translate `text` if not throttled and changed; else None (no update)."""
        text = (text or "").strip()
        if not text:
            return None
        if text == self._last_src:
            return None
        if (self._clock() - self._last_at) < self._interval and self._last_src is not None:
            return None
        out = self._translate(text)
        self._last_at = self._clock()
        self._last_src = text
        self._last_out = out
        return out

    def final(self, text: str) -> Optional[str]:
        """Force a translation of the committed text, ignoring throttle."""
        text = (text or "").strip()
        if not text:
            return None
        out = self._translate(text)
        self._last_at = self._clock()
        self._last_src = text
        self._last_out = out
        return out


class NllbTranslator:
    """CTranslate2 + NLLB-200 text translator.

    Construct via `NllbTranslator.load(model_dir, device)` for production, or
    pass `translator`/`tokenizer` directly (tests). The NLLB recipe: set the
    tokenizer src_lang, encode → subword tokens, translate_batch with the target
    language as the decoder target_prefix, then decode the output tokens.
    """
    def __init__(self, translator, tokenizer):
        self._ct2 = translator
        self._tok = tokenizer

    @classmethod
    def load(cls, model_dir, device: str = "cpu", compute_type: str = "int8"):  # pragma: no cover - native
        import ctranslate2
        from transformers import AutoTokenizer
        translator = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        return cls(translator, tokenizer)

    def translate(self, text: str, src: str, tgt: str) -> str:
        self._tok.src_lang = src
        tokens = self._tok.convert_ids_to_tokens(self._tok.encode(text))
        results = self._ct2.translate_batch([tokens], target_prefix=[[tgt]])
        out_tokens = results[0].hypotheses[0][1:]  # drop the leading target-lang token
        out_ids = self._tok.convert_tokens_to_ids(out_tokens)
        return self._tok.decode(out_ids, skip_special_tokens=True)
