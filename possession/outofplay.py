"""Out-of-play detector (T): dead ball while the ball is tracked. No learning, no labels.

Pre-registered in docs/worklog.md ("2026-11-15 (out-of-play detector, T)"); constants copied from there.

  out        ball tracked more than 0.5 m beyond a touchline or goal line on >= 10 consecutive frames
  dead from  the first frame of that run
  until      the first frame >= 2 s after the run ends with the ball tracked inside the pitch (0.5 m margin) and
             ball speed (centred 10-frame median of frame-to-frame speed) >= 6 m/s; at most 60 s after the start
"""

import numpy as np

FPS = 25
OUT_MARGIN_M = 0.5
OUT_MIN_FRAMES = 10
SPEED_MEDIAN_FRAMES = 10
RESTART_MPS = 6.0
RESTART_DELAY_S = 2
MAX_S = 60


def ball_speed(ball):
    """Centred 10-frame median of |ball(t) - ball(t-1)| x 25; NaN where untracked."""
    raw = np.full(len(ball), np.nan)
    raw[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * FPS
    k = SPEED_MEDIAN_FRAMES
    pad = np.r_[np.full(k // 2, np.nan), raw, np.full(k - k // 2 - 1, np.nan)]
    win = np.lib.stride_tricks.sliding_window_view(pad, k)
    out = np.full(len(ball), np.nan)
    have = np.isfinite(win).any(axis=1)
    out[have] = np.nanmedian(win[have], axis=1)
    return out


def outside(ball, pitch, margin=OUT_MARGIN_M):
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def inside(ball, pitch, margin=OUT_MARGIN_M):
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x >= -margin) & (x <= length + margin) & (y >= -margin) & (y <= width + margin)


def stretches(ball, pitch, halves):
    """[(start, end)] inclusive dead stretches inside the (first, last) halves."""
    out = outside(ball, pitch)
    ins = inside(ball, pitch)
    v = ball_speed(ball)
    restart = ins & (v >= RESTART_MPS)
    result = []
    for first, last in halves:
        d = np.diff(np.r_[0, out[first:last + 1].astype(int), 0])
        runs = [(first + s, first + e - 1) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])
                if e - s >= OUT_MIN_FRAMES]
        for s, e in runs:
            if result and s <= result[-1][1]:
                continue                                    # already inside a stretch
            lo = e + 1 + RESTART_DELAY_S * FPS
            hi = min(s + MAX_S * FPS, last)
            idx = np.where(restart[lo:hi + 1])[0] if lo <= hi else np.array([], dtype=int)
            end = lo + idx[0] - 1 if len(idx) else hi
            result.append((s, max(end, e)))
    return result


def flags(ball, pitch, halves, n=None):
    f = np.zeros(n or len(ball), dtype=bool)
    for s, e in stretches(ball, pitch, halves):
        f[s:e + 1] = True
    return f
