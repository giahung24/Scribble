"""LocalAgreement-2 commit policy for streaming Whisper.

The streaming ASR re-decodes a growing audio buffer every ~Nms, producing a
fresh hypothesis each time. A word is only trustworthy once it stops changing:
we commit the longest common prefix (by word text) of the two most recent
hypotheses, beyond what is already committed. The uncommitted tail is the live
partial. This is the UFAL whisper_streaming algorithm.

Words are (text, start_ms, end_ms) tuples so callers keep timing for the UI.
"""
from __future__ import annotations

Word = tuple[str, int, int]


class LocalAgreement:
    def __init__(self) -> None:
        self.committed: list[Word] = []
        self._prev: list[Word] = []
        self._latest: list[Word] = []

    def add(self, hypothesis: list[Word]) -> list[Word]:
        """Feed a fresh full hypothesis. Returns the words newly committed."""
        self._latest = list(hypothesis)
        n = len(self.committed)
        prev_tail = self._prev[n:]
        new_tail = hypothesis[n:]
        agreed: list[Word] = []
        for a, b in zip(prev_tail, new_tail):
            if a[0] == b[0]:
                agreed.append(b)
            else:
                break
        self.committed.extend(agreed)
        self._prev = list(hypothesis)
        return agreed

    def partial(self) -> list[Word]:
        """Uncommitted tail of the most recent hypothesis (the live preview)."""
        return self._latest[len(self.committed):]

    def reset(self) -> None:
        self.committed = []
        self._prev = []
        self._latest = []
