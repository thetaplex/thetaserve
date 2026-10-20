"""Private, atomic file writes shared by state and secrets."""

import os
import tempfile
from pathlib import Path


def write_private(path: Path, text: str) -> None:
    """Write `text` to `path` atomically, readable only by the owner on POSIX.

    A crash mid-write leaves the old file intact. On Windows the mode is ignored and the
    file inherits the user profile's ACL, which already restricts it to the user.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).chmod(0o600)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
