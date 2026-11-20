"""Drop-in for the source repo's ``RemoteZip`` that reads a local archive."""

from __future__ import annotations

import threading
import zipfile
from pathlib import Path


class LocalZip:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._local = threading.local()
        self.length = Path(path).stat().st_size
        with zipfile.ZipFile(self._path) as archive:
            self.entries = {i.filename: i for i in archive.infolist() if not i.is_dir()}

    def read(self, name: str) -> bytes:
        archive = getattr(self._local, "zip", None)
        if archive is None:  # ZipFile handles are not thread-safe
            archive = self._local.zip = zipfile.ZipFile(self._path)
        return archive.read(name)
