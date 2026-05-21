"""Compatibility helpers for older local verl checkouts."""

from __future__ import annotations

import os
from pathlib import Path


def copy_local_path_from_hdfs(src: str, cache_dir: str | None = None) -> str:
    """Best-effort local-path resolver used when verl.utils.fs is unavailable.

    The local training setups in this workspace pass ordinary filesystem paths or
    HuggingFace model IDs. For those cases we should return the path unchanged
    after expanding ``~``. HDFS URIs are not supported in this compatibility
    fallback.
    """
    src = os.path.expanduser(str(src))
    if src.startswith('hdfs://'):
        raise NotImplementedError(
            'HDFS paths require verl.utils.fs, which is unavailable in the current local verl checkout.'
        )
    if cache_dir is not None:
        Path(os.path.expanduser(cache_dir)).mkdir(parents=True, exist_ok=True)
    return src
