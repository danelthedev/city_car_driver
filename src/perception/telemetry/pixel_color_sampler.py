"""
Samples specific screen pixels and checks whether their color is close to a
target color.  Uses mss for fast, zero-dependency screen capture.
"""

import math
import time


_DEFAULT_PIXELS = ((2286, 80), (2324, 80))


def _color_distance(bgr1: tuple[int, int, int], bgr2: tuple[int, int, int]) -> float:
    """Euclidean distance in BGR space."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(bgr1, bgr2)))


class PixelColorSampler:
    def __init__(
        self,
        pixels: tuple[tuple[int, int], ...] = _DEFAULT_PIXELS,
        target_hex: str = "#D2B819",
        threshold: float = 25.0,
        poll_interval_ms: float = 50.0,
    ):
        """
        Parameters
        ----------
        pixels          : sequence of (x, y) screen coordinates to sample
        target_hex      : hex color string to match (e.g. '#D2B819')
        threshold       : max Euclidean distance in RGB space to count as a match
        poll_interval_ms: minimum time between samples
        """
        self.pixels = tuple(pixels)
        self.threshold = float(threshold)
        self.poll_interval_s = max(0.0, float(poll_interval_ms) / 1000.0)

        hex_clean = target_hex.lstrip("#")
        r = int(hex_clean[0:2], 16)
        g = int(hex_clean[2:4], 16)
        b = int(hex_clean[4:6], 16)
        self._target_bgr: tuple[int, int, int] = (b, g, r)

        self._sct = None
        self._next_poll_ts = 0.0
        # last result: list of (x, y, matched, distance) — one per pixel
        self.last_results: list[tuple[int, int, bool, float]] = []

    def _ensure_sct(self):
        if self._sct is None:
            mss = __import__("mss")
            self._sct = mss.mss()

    def sample_if_due(self, now_ts: float | None = None) -> list[tuple[int, int, bool, float]] | None:
        """
        Sample pixels if the poll interval has elapsed.

        Returns a list of (x, y, matched, distance) tuples, or None if it was
        not yet time to sample.  The result is also stored in self.last_results.
        """
        ts = time.perf_counter() if now_ts is None else float(now_ts)
        if ts < self._next_poll_ts:
            return None

        self._next_poll_ts = ts + self.poll_interval_s

        try:
            self._ensure_sct()
            results = []
            for x, y in self.pixels:
                region = {"left": x, "top": y, "width": 1, "height": 1}
                shot = self._sct.grab(region)
                # mss raw bytes are BGRA on Windows
                raw = shot.raw
                b, g, r = raw[0], raw[1], raw[2]
                dist = _color_distance((b, g, r), self._target_bgr)
                results.append((x, y, dist <= self.threshold, round(dist, 1), (r, g, b)))
            self.last_results = results
            return results
        except Exception:
            return None

    def close(self):
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
