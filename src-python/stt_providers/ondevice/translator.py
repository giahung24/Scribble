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
