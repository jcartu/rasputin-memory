from __future__ import annotations

import fcntl
import os


def acquire_lock(name: str) -> int:
    lock_path = f"/tmp/rasputin_{name}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd
