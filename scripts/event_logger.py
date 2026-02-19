from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlEventLogger:
    def __init__(self, path: str):
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = log_path.open("w", encoding="utf-8")

    def log(self, event_type: str, **payload: Any) -> None:
        record = {"event": event_type, **payload}
        self._fp.write(json.dumps(record, sort_keys=True) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()
