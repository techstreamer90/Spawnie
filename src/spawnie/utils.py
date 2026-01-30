"""Utility functions for Spawnie."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """
    Write JSON data to file atomically using a temp file and rename.

    This prevents corruption if the process is interrupted mid-write.

    Args:
        path: Target file path
        data: Data to serialize as JSON
        indent: JSON indentation (default 2)
    """
    # Create temp file in same directory for atomic rename
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        # Atomic rename
        Path(tmp_path).replace(path)
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """
    Write text to file atomically using a temp file and rename.

    Args:
        path: Target file path
        content: Text content to write
    """
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp_path).replace(path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
