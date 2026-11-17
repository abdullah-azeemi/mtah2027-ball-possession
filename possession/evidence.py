"""Evidence-layer variants: which team has the ball while it is in play.

Pre-registered in docs/worklog.md ("2026-11-15 (evidence layer on sample matches)"). Every function takes
arrays (pos (n, P, 2) m, ball (n, 2) m, team (P,)) and works on any match, ours or Metrica's samples.

Per-frame evidence codes: 0 home, 1 away, -1 no vote.

    nearest_codes     team of the nearest tracked player, any distance (T_E / reference)
    controlled_codes  exactly one team within RADIUS_M (the submission)
    slow_ball         ball speed under FLIGHT_SPEED_MS: a faster ball is in flight and votes for nobody
    with_fallback     where no controlled vote lies within FALLBACK_FRAMES, a slow ball votes for the nearest team

Segmentation of one half (returns 0/1 per frame, -1 only if the half has no vote at all):

    vote_segment      centred majority vote, ties to the previous decision, then hold (submit.majority + submit.hold)
    reception_segment switch on reception: runs of same-team votes shorter than MIN_RUN_VOTES are dropped as
                      deflections; a possession starts at the first vote of each kept run and is held until the next
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import submit  # noqa: E402

FPS = 25
RADIUS_M = submit.CONTROL_RADIUS_M      # 1.5
VOTE_WINDOW = submit.VOTE_WINDOW        # 60
FLIGHT_SPEED_MS = submit.FLIGHT_SPEED_MS   # 7.0, speed over 0.16 s
FALLBACK_FRAMES = 75                   # 3 s
MIN_RUN_VOTES = 5                      # 0.2 s of control


def nearest_codes(pos, ball, team, chunk=20000):
    n = len(ball)
    codes = np.full(n, -1, dtype=np.int8)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        d2 = ((pos[sl] - ball[sl][:, None, :]) ** 2).sum(axis=2)
        d2 = np.where(np.isnan(d2), np.inf, d2)
        nearest = d2.argmin(axis=1)
        ok = np.isfinite(d2.min(axis=1))
        codes[sl] = np.where(ok, team[nearest], -1)
    return codes


def controlled_codes(pos, ball, team, radius=RADIUS_M):
    return submit.controlled_codes(pos, ball, team, radius)


ball_speed = submit.ball_speed


def slow_ball(codes, ball, limit=FLIGHT_SPEED_MS):
    return submit.drop_in_flight(codes, ball, limit)


def with_fallback(primary, fallback, frames=FALLBACK_FRAMES):
    """primary where a primary vote lies within +/-frames; otherwise the fallback code."""
    has = np.concatenate(([0], np.cumsum(primary >= 0)))
    idx = np.arange(len(primary))
    low = np.maximum(0, idx - frames)
    high = np.minimum(len(primary), idx + frames + 1)
    near = (has[high] - has[low]) > 0
    return np.where(near, primary, fallback).astype(np.int8)


def vote_segment(codes, window=VOTE_WINDOW, ties="previous"):
    if ties == "home":
        h = np.concatenate(([0], np.cumsum(codes == 0)))
        a = np.concatenate(([0], np.cumsum(codes == 1)))
        idx = np.arange(len(codes))
        low, high = np.maximum(0, idx - window), np.minimum(len(codes), idx + window + 1)
        hv, av = h[high] - h[low], a[high] - a[low]
        vote = np.where(hv + av > 0, np.where(hv >= av, 0, 1), -1).astype(np.int8)
    else:
        vote = submit.majority(codes, window)
    return submit.hold(vote)


def reception_segment(codes, min_votes=MIN_RUN_VOTES):
    idx = np.flatnonzero(codes >= 0)
    out = np.full(len(codes), -1, dtype=np.int8)
    if len(idx) == 0:
        return out
    teams = codes[idx]
    breaks = np.flatnonzero(np.diff(teams) != 0) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(idx)]))
    kept = [(idx[s], teams[s]) for s, e in zip(starts, ends) if e - s >= min_votes]
    if not kept:                           # no sustained control anywhere: fall back to the vote
        return vote_segment(codes)
    marks = np.full(len(codes), -1, dtype=np.int8)
    for frame, tm in kept:
        marks[frame] = tm
    return submit.hold(marks)


VARIANTS = ("N", "C", "CS", "CSF", "CSR", "CSFR")


def evidence(variant, pos, ball, team):
    if variant == "N":
        return nearest_codes(pos, ball, team)
    c = controlled_codes(pos, ball, team)
    if variant == "C":
        return c
    cs = slow_ball(c, ball)
    if "F" in variant:
        return with_fallback(cs, slow_ball(nearest_codes(pos, ball, team), ball))
    return cs


def team_labels(variant, pos, ball, team, halves):
    """0/1 per frame inside the halves, -1 outside. halves: [(first, last)] inclusive."""
    codes = evidence(variant, pos, ball, team)
    out = np.full(len(ball), -1, dtype=np.int8)
    for first, last in halves:
        seg = codes[first:last + 1]
        if variant == "N":
            out[first:last + 1] = vote_segment(seg, window=50, ties="home")
        elif variant.endswith("R"):
            out[first:last + 1] = reception_segment(seg)
        else:
            out[first:last + 1] = vote_segment(seg)
    return out
