import time, json
from pathlib import Path
from contextlib import contextmanager
from typing import Any
from datetime import datetime

class Timer:
    def __init__(self, 
                out_path = Path("evals", f"timings_{datetime.now().strftime("%H%M_%d%m-%Y")}.json"), 
                enabled: bool = False):
        self.enabled = enabled
        self.out_path = Path(out_path)
        self.started_at = datetime.now().isoformat()
        self.timings: dict[str, float] = {}

    @contextmanager
    def start_timer(self, key: str):
        """Context manager that only measures time if enabled."""
        if not self.enabled:
            # no-op context manager
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] = round(time.perf_counter() - start, 4)

    def save(self):
        """Write timings to disk only if enabled."""
        if not self.enabled:
            return

        self.out_path.parent.mkdir(exist_ok=True, parents=True)
        # if self.out_path.exists():
        #     existing = json.loads(self.out_path.read_text())
        # else:
        #     existing = []

        # existing.append(run)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(self.timings, f, indent=4)
            