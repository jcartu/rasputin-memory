from __future__ import annotations

import importlib
from types import ModuleType


def safe_import(primary: str, fallback: str) -> ModuleType:
    try:
        return importlib.import_module(primary)
    except ModuleNotFoundError:
        return importlib.import_module(fallback)
