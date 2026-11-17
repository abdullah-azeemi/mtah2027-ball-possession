"""Write the submission: possession from controlled-ball evidence, the loose ball left with the last team.

    python possession/submit.py --out <team_name>.csv

Labels every video frame H, A, D or X from the provided tactical tracking alone. Four parts: per-frame team
evidence, a vote, a hold, and dead-ball rules.

Evidence
--------
The reference baseline gives every frame with a tracked ball to the team of the nearest tracked slot,
however far away that player is. The challenge's own possession rule says (CHALLENGE.md, "Ball in play"):

    "In phases where the ball is contested or loose, possession stays with the last team that had it."

A frame where the nearest player is 8 m from the ball is not evidence that this team controls it; it
is a loose ball. So here a frame casts a vote only when exactly one team has a tracked player within
CONTROL_RADIUS_M of the ball -- i.e. the ball is under control and not contested. Contested, loose
and untracked-ball frames cast no vote and are filled by HOLD, which is literally "possession stays
with the last team that had it".

Vote
----
A centred majority vote over +/- VOTE_WINDOW frames of the gated votes. Ties mean the window holds
no controlled evidence either way, so they keep the previous decided possession (the reference
baseline breaks ties to home instead, which silently hands every contest to Barcelona).

The window is wider than the reference's 50 because gating removes roughly half of the frames from
the vote; +/-60 gated frames carry about the evidence of +/-50 ungated ones.

Ball in flight
--------------
A ball moving faster than FLIGHT_SPEED_MS (7 m/s, centred over 0.16 s) is a pass, a long ball or a
clearance, not a ball at someone's feet; a defender it flies past (or over) within 1.5 m is not in
control. Such frames cast no vote.

Both evidence rules were tested, pre-registered, on Metrica's open sample games 1 and 2 against
event-derived possession (docs/results/evidence_layer.md): spellF1 0.575 / 0.635, against 0.521 / 0.488
for the nearest-player rule.

Hold
----
Frames with no vote keep the previous label inside each half; the head of a half is back-filled with
its first vote.

Dead ball
---------
Stoppage detector (players' rolling median speed < 1.0 m/s with the ball not moving, for >= 5 s),
the second-half pre-kick-off frames 110808-113741, ball-untracked runs of 10 s or more, and the
out-of-play detector, are written `D`.

Everything outside the two halves is `X` (never scored).

No randomness and no learning: the same tracking gives the same bytes.
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import outofplay  # noqa: E402
import tracking  # noqa: E402
import v1  # noqa: E402  (dead_flags and its constants)
import v1b  # noqa: E402  (long_gap_flags)
from games import GAME, half_video_blocks  # noqa: E402

CONTROL_RADIUS_M = 1.5          # a player this close to the ball is controlling it, not chasing it
VOTE_WINDOW = 60                # half-width of the centred majority vote, in frames
FLIGHT_SPEED_MS = 7.0           # a faster ball is in flight: no frame vote
SPEED_HALF_SPAN = 2             # ball speed from ball[t+2] - ball[t-2] (0.16 s at 25 Hz)
FPS = 25
N_FRAMES = 199319
LABELS = {"H", "A", "D", "X"}


def controlled_codes(pos, ball, team, radius=CONTROL_RADIUS_M, chunk=20000):
    """Team code per frame (0 home, 1 away, -1 no controlled evidence).

    A frame gets a code only when exactly one team has a tracked slot within `radius` metres of the
    ball; that team is the code. The ball is untracked, nobody is within the radius, or both teams are within the
    radius (a contest) -> -1. Distances are Euclidean in metres, as in baseline.nearest_team.
    """
    n = len(ball)
    codes = np.full(n, -1, dtype=np.int8)
    r2 = float(radius) ** 2
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        dx = pos[sl, :, 0] - ball[sl, 0][:, None]
        dy = pos[sl, :, 1] - ball[sl, 1][:, None]
        d2 = dx * dx + dy * dy
        d2 = np.where(np.isnan(d2), np.inf, d2)
        nearest = d2.argmin(axis=1)
        home = np.where(team[None, :] == 0, d2, np.inf).min(axis=1)
        away = np.where(team[None, :] == 1, d2, np.inf).min(axis=1)
        found = (home <= r2) ^ (away <= r2)                     # exactly one team in control range
        codes[sl] = np.where(found, team[nearest], -1)
    return codes


def majority(codes, window=VOTE_WINDOW):
    """Centred majority vote. -1 where the window holds no vote; ties keep the previous decision."""
    home_c = np.concatenate(([0], np.cumsum(codes == 0)))
    away_c = np.concatenate(([0], np.cumsum(codes == 1)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_c[high] - home_c[low]
    away = away_c[high] - away_c[low]
    seen = (home + away) > 0
    out = np.full(len(codes), -1, dtype=np.int8)
    out[seen] = np.where(home[seen] >= away[seen], 0, 1)

    # A tie means the window holds no controlled evidence in either direction. Keep the previous
    # decided possession; at the head of the sequence use the next decided one.
    last = -1
    for t in range(len(codes)):
        if not seen[t]:
            continue
        if home[t] == away[t]:
            if last < 0:
                j = t
                while j < len(codes) and not (seen[j] and home[j] != away[j]):
                    j += 1
                last = int(out[j]) if j < len(codes) else 0
            out[t] = last
        else:
            last = int(out[t])
    return out


def hold(segment):
    """Fill -1 with the previous known vote; at the head of the segment use the first known vote."""
    n = len(segment)
    last_known = np.maximum.accumulate(np.where(segment >= 0, np.arange(n), -1))
    filled = np.where(last_known >= 0, segment[np.maximum(last_known, 0)], -1)
    if (filled < 0).any() and (segment >= 0).any():
        filled[filled < 0] = segment[segment >= 0][0]
    return filled


def dead_flags(speed, ball, graded):
    """T_E's dead-ball frames: stoppages, second-half pre-kick-off, long gaps and out of play."""
    dead = v1.dead_flags(speed, ball, graded)
    dead |= v1b.long_gap_flags(ball, graded)
    halves = [(first, last) for _, first, last in half_video_blocks(GAME)]
    dead |= outofplay.flags(ball, tracking.PITCH, halves) & graded
    return dead & graded


def ball_speed(ball, half_span=SPEED_HALF_SPAN):
    """Centred ball speed in m/s; NaN where either end is untracked or outside the array."""
    n = len(ball)
    out = np.full(n, np.nan)
    k = half_span
    if n > 2 * k:
        step = ball[2 * k:] - ball[:-2 * k]
        out[k:n - k] = np.hypot(step[:, 0], step[:, 1]) * FPS / (2 * k)
    return out


def drop_in_flight(codes, ball, limit=FLIGHT_SPEED_MS):
    """No vote where the ball is in flight (faster than `limit`) or its speed is unknown."""
    out = codes.copy()
    out[~(ball_speed(ball) < limit)] = -1
    return out


def make_labels(pos=None, ball=None, team=None, speed=None,
                radius=CONTROL_RADIUS_M, window=VOTE_WINDOW, flight_speed=FLIGHT_SPEED_MS):
    """Labels 'H', 'A', 'D', 'X' for every video frame. Loads the tracking when not given.

    flight_speed=None switches the in-flight gate off (the original controlled-ball rule)."""
    if pos is None:
        pos, ball, team, speed = tracking.load_tracking(with_speed=True)
    n = len(ball)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    codes = controlled_codes(pos, ball, team, radius)
    if flight_speed is not None:
        codes = drop_in_flight(codes, ball, flight_speed)
    dead = dead_flags(speed, ball, graded)

    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        # Vote inside each half only: the window and the tie memory must not reach across half-time.
        segment = hold(majority(codes[first:last + 1], window))
        labels[first:last + 1] = np.where(segment == 0, "H",
                                          np.where(segment == 1, "A", "D"))
    labels[dead] = "D"
    return labels


def write_submission(path, labels):
    """`frame,label`, one row per video frame, LF line endings on every platform."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    frame.to_csv(path, index=False, lineterminator="\n")


def check_format(path):
    """The file contract of SUBMISSION.md, checked again after writing."""
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["frame", "label"], "header must be frame,label"
    assert len(frame) == N_FRAMES, "expected {} rows, found {}".format(N_FRAMES, len(frame))
    assert (frame["frame"].values == np.arange(N_FRAMES)).all(), "frame must run 0..{} in order".format(N_FRAMES - 1)
    assert set(frame["label"].unique()) <= LABELS, "labels must be H, A, D or X"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(HERE, "output", "submission", "submission.csv"),
                        help="where to write the CSV (name it <team_name>.csv)")
    parser.add_argument("--radius", type=float, default=CONTROL_RADIUS_M,
                        help="control radius in metres (default %(default)s)")
    parser.add_argument("--flight-speed", type=float, default=FLIGHT_SPEED_MS,
                        help="ball faster than this (m/s) is in flight and casts no vote; 0 disables "
                             "(default %(default)s)")
    parser.add_argument("--window", type=int, default=VOTE_WINDOW,
                        help="half-width of the majority vote in frames (default %(default)s)")
    args = parser.parse_args(argv)

    print("loading tracking ...")
    pos, ball, team, speed = tracking.load_tracking(with_speed=True)
    labels = make_labels(pos, ball, team, speed, args.radius, args.window,
                         None if args.flight_speed <= 0 else args.flight_speed)

    write_submission(args.out, labels)
    check_format(args.out)
    counts = {c: int((labels == c).sum()) for c in "HADX"}
    print("radius {:.2f} m, vote window +/-{} frames, in-flight gate {} m/s".format(
        args.radius, args.window, args.flight_speed if args.flight_speed > 0 else "off"))
    print("wrote {}".format(args.out))
    print("rows {}  H {}  A {}  D {}  X {}".format(len(labels), *[counts[c] for c in "HADX"]))
    print("sha256 {}".format(sha256(args.out)))
    print("validate with: python evaluation/validate_submission.py {}".format(args.out))


if __name__ == "__main__":
    main()
