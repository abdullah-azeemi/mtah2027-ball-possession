"""Simple, model-free ways to fill a ball gap. These are the bar a learned model must clear.

Each function takes the ball the method may see, ``shown`` (WINDOW, 2) in metres with NaN where it
is hidden or really missing, plus the gap [gap_start, gap_end) of the trial, and returns the guessed
positions for the gap steps, shape (gap_end - gap_start, 2).

* last observation carried forward (LOCF): the ball stays where it was last seen. Uses the past only.
* constant velocity: continues from the last two tracked positions at that velocity, clipped to
  the pitch. Past only; this is the "use the earlier frames' direction and speed" idea.
* linear interpolation: a straight line between the last tracked position before the gap and the
  first one after it. Uses both sides, so it is the natural two-sided baseline.
"""

import numpy as np

from data import PITCH_L, PITCH_W


def _last_two_before(shown, gap_start):
    """Indices of the last two tracked steps before gap_start (None if there are not two)."""
    seen = np.flatnonzero(np.isfinite(shown[:gap_start, 0]))
    return (seen[-2], seen[-1]) if len(seen) >= 2 else (None, seen[-1] if len(seen) else None)


def _first_after(shown, gap_end):
    seen = np.flatnonzero(np.isfinite(shown[gap_end:, 0]))
    return gap_end + seen[0] if len(seen) else None


def locf(shown, gap_start, gap_end):
    _, last = _last_two_before(shown, gap_start)
    if last is None:
        return np.full((gap_end - gap_start, 2), np.nan)
    return np.repeat(shown[last][None], gap_end - gap_start, axis=0)


def constant_velocity(shown, gap_start, gap_end):
    before, last = _last_two_before(shown, gap_start)
    if last is None:
        return np.full((gap_end - gap_start, 2), np.nan)
    velocity = (shown[last] - shown[before]) / (last - before) if before is not None else 0.0
    steps = np.arange(gap_start, gap_end) - last
    guess = shown[last][None] + steps[:, None] * velocity
    guess[:, 0] = np.clip(guess[:, 0], 0, PITCH_L)
    guess[:, 1] = np.clip(guess[:, 1], 0, PITCH_W)
    return guess


def linear_interpolation(shown, gap_start, gap_end):
    _, last = _last_two_before(shown, gap_start)
    after = _first_after(shown, gap_end)
    if last is None or after is None:
        return locf(shown, gap_start, gap_end)
    weight = (np.arange(gap_start, gap_end) - last) / float(after - last)
    return shown[last][None] * (1 - weight[:, None]) + shown[after][None] * weight[:, None]


BASELINES = {
    "last position (past only)": locf,
    "constant velocity (past only)": constant_velocity,
    "linear interpolation (two-sided)": linear_interpolation,
}
