"""On-device model cache + download-on-demand manager.

Resolves model directories under a root (default ~/.voicescribe/models) and
fetches models on first use. The actual network fetch is injected as
`downloader(kind, name, dest)` so it is testable; the default uses
huggingface_hub.snapshot_download (added with the runtime deps).

is_*_present treats a non-empty directory as downloaded (good enough; a partial
download is cleaned by callers on failure)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

NLLB_REPO = "facebook/nllb-200-distilled-600M"


def _default_root() -> Path:
    base = os.environ.get("VOICESCRIBE_HOME")
    root = Path(base) if base else (Path.home() / ".voicescribe")
    return root / "models"


def _hf_download(kind: str, name: str, dest: Path) -> None:  # pragma: no cover - network
    from huggingface_hub import snapshot_download
    repo = NLLB_REPO if kind == "nllb" else f"Systran/faster-whisper-{name}"
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(dest))


class ModelManager:
    def __init__(self, root: Optional[Path] = None,
                 downloader: Callable[[str, str, Path], None] = _hf_download):
        self.root = Path(root) if root else _default_root()
        self._download = downloader

    def whisper_dir(self, size: str) -> Path:
        return self.root / "whisper" / size

    def nllb_dir(self) -> Path:
        return self.root / "nllb-200-distilled-600M"

    @staticmethod
    def _present(d: Path) -> bool:
        return d.is_dir() and any(d.iterdir())

    def is_whisper_present(self, size: str) -> bool:
        return self._present(self.whisper_dir(size))

    def is_nllb_present(self) -> bool:
        return self._present(self.nllb_dir())

    def ensure_whisper(self, size: str) -> Path:
        d = self.whisper_dir(size)
        if not self._present(d):
            self._download("whisper", size, d)
        return d

    def ensure_nllb(self) -> Path:
        d = self.nllb_dir()
        if not self._present(d):
            self._download("nllb", "nllb-200-distilled-600M", d)
        return d
