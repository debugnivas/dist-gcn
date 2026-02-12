"""
Story 3.1: High-resolution timer utility.

Context-manager and dictionary-based timer using time.perf_counter
for nanosecond-precision monotonic timing.
"""
import time
from typing import Dict, Optional


class Timer:
    """
    High-resolution timer for profiling code sections.

    Usage:
        timer = Timer()
        timer.start('forward')
        # ... code ...
        timer.stop('forward')
        print(timer.get('forward'))  # seconds

    Or as context manager:
        with timer.section('forward'):
            # ... code ...
    """

    def __init__(self):
        self._starts: Dict[str, float] = {}
        self._durations: Dict[str, float] = {}

    def start(self, name: str):
        """Start timing a named section."""
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing a named section. Returns duration in seconds."""
        if name not in self._starts:
            raise ValueError(f"Timer '{name}' was not started.")
        duration = time.perf_counter() - self._starts[name]
        self._durations[name] = self._durations.get(name, 0.0) + duration
        del self._starts[name]
        return duration

    def get(self, name: str) -> float:
        """Get accumulated duration for a named section (seconds)."""
        return self._durations.get(name, 0.0)

    def reset(self, name: Optional[str] = None):
        """Reset timer(s)."""
        if name:
            self._durations.pop(name, None)
            self._starts.pop(name, None)
        else:
            self._durations.clear()
            self._starts.clear()

    def all_durations(self) -> Dict[str, float]:
        """Return all recorded durations."""
        return dict(self._durations)

    class _Section:
        """Context manager for a timer section."""
        def __init__(self, timer, name):
            self.timer = timer
            self.name = name

        def __enter__(self):
            self.timer.start(self.name)
            return self

        def __exit__(self, *args):
            self.timer.stop(self.name)

    def section(self, name: str):
        """Context manager for timing a section."""
        return self._Section(self, name)
