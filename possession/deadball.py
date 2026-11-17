"""Dead-ball (stoppage) detector from the tracking alone. No learning, no labels.

Why it exists: the reference baseline writes D wherever the ball is not tracked. That misses most
dead ball (the ball is often tracked while dead) and marks live play as dead (worklog findings 31, 32).
What the tracking does show around a stoppage (findings 35 to 38): the players slow down together, and
the ball is mostly not tracked or not moving.

Rule (parameters theta and T are chosen by the pre-registered procedure in docs/worklog.md)
  s(t)        3 s centred rolling mean of the median speed of the tracked slots (the tracking's speed column)
  candidate   s(t) < theta  AND  the ball is not moving, meaning: not tracked, or displaced less than 1 m
              between t - 1 s and t + 1 s. A ball that is tracked but whose neighbours 1 s away are not
              tracked is treated as moving: stillness needs evidence.
  stoppage    a maximal run of candidates that lasts at least T seconds; runs separated by less than 2 s
              are merged first (a ball that flickers into view must not cut a stoppage in two)
Only graded frames can be candidates: half-time and the time outside the match are not play.
"""

import warnings

import numpy as np

FPS = 25
ROLL_S = 3          # smoothing of the players' speed
STILL_WINDOW_S = 2  # window over which the ball's displacement is measured
STILL_MAX_M = 1.0
MERGE_S = 2


def rolling_player_speed(speed, window_s=ROLL_S):
    """Per frame: centred rolling mean over `window_s` of the median speed of the tracked slots."""
    with warnings.catch_warnings():             # frames with no tracked slot give NaN, which is intended
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(np.isfinite(speed), speed, np.nan), axis=1)
    n = len(median)
    k = int(window_s * FPS)
    total = np.concatenate([[0.0], np.cumsum(np.nan_to_num(median))])
    count = np.concatenate([[0], np.cumsum(np.isfinite(median))])
    index = np.arange(n)
    low, high = np.maximum(0, index - k // 2), np.minimum(n, index + k // 2 + 1)
    have = count[high] - count[low]
    return np.where(have > 0, (total[high] - total[low]) / np.maximum(have, 1), np.nan)


def ball_not_moving(ball, window_s=STILL_WINDOW_S, max_move_m=STILL_MAX_M):
    """Boolean per frame: ball not tracked, or displaced under `max_move_m` over the window."""
    n = len(ball)
    visible = np.isfinite(ball[:, 0])
    half = int(window_s * FPS) // 2
    moved = np.full(n, np.nan)
    both = visible[:-2 * half] & visible[2 * half:]
    distance = np.hypot(*(ball[2 * half:] - ball[:-2 * half]).T)
    moved[half:n - half][both] = distance[both]
    still = np.where(np.isfinite(moved), moved < max_move_m, False)
    return ~visible | still


def runs(mask):
    """[(start, end)] of the maximal runs of True, end exclusive."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    change = np.diff(padded)
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def stoppages(candidate, min_seconds, merge_seconds=MERGE_S):
    """Stoppages as [(start, end)] frames (end exclusive) from a boolean candidate array."""
    merged = []
    for start, end in runs(candidate):
        if merged and start - merged[-1][1] < merge_seconds * FPS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(s, e) for s, e in merged if e - s >= min_seconds * FPS]


def detect(speed, ball, graded, theta, min_seconds):
    """(stoppages, per-frame boolean dead flag) for one parameter setting."""
    candidate = (rolling_player_speed(speed) < theta) & ball_not_moving(ball) & graded
    found = stoppages(candidate, min_seconds)
    flag = np.zeros(len(ball), dtype=bool)
    for start, end in found:
        flag[start:end] = True
    return found, flag
