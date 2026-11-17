"""Regenerate the submission file from the provided data.

    python possession/make_submission.py --out <team_name>.csv

Writes the possession file (header `frame,label`, 199,319 rows) for the chosen candidate, T_E ("v1f"): variant E
of possession/v1b.py plus the out-of-play detector of possession/outofplay.py:

  1. on each frame with the ball tracked, the team of the nearest tracked slot;
  2. a centred majority vote over +/- 50 frames (ties to home);
  3. where the window holds no vote (ball not tracked), the previous vote is held;
  4. D on the frames flagged by the stoppage detector (players' rolling median speed below 1.0 m/s with the
     ball not moving, for at least 5 s) and on the second-half frames 110808-113741 (FAQ.md: all D);
  5. D on every run of 250 or more consecutive frames (10 s) where the ball is not tracked (a long gap is
     mostly dead ball: worklog "gap check, v1c");
  6. D while the ball is out of play: from the ball tracked more than 0.5 m beyond a touchline or goal line for
     0.4 s, until a restart (ball back on the pitch moving at 6 m/s or more, at least 2 s later; 60 s at most)
     (worklog "out-of-play detector, T" and "adoption of T_E");
  7. X outside the two halves (never scored).

The other candidates (F gap scorecard, G learned gap model) are documented in docs/results/ and are not used here.

The pipeline has no randomness and no learning, so the same data gives the same bytes. The SHA-256 of the file
is printed; docs/REPRODUCE.md lists the value expected for the data bundle used in development.

Needs numpy and pandas only. The data must be in ./data (or MTAH_DATA_DIR), see DATA.md.
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

import baseline  # noqa: E402
import outofplay  # noqa: E402
import tracking  # noqa: E402
import v1  # noqa: E402  (dead_flags and its constants)
import v1b  # noqa: E402  (votes, hold, build, WINDOW)
from games import GAME, half_video_blocks  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "output", "submission", "possession_v1c.csv")
N_FRAMES = 199319
LABELS = {"H", "A", "D", "X"}


def make_labels():
    """The label array (one letter per video frame): steps 1 to 7 above."""
    pos, ball, team, speed = tracking.load_tracking(with_speed=True)
    halves = [(first, last) for _, first, last in half_video_blocks(GAME)]
    graded = np.zeros(len(ball), dtype=bool)
    for first, last in halves:
        graded[first:last + 1] = True
    codes = baseline.nearest_team(pos, ball, team)
    vote = v1b.votes(codes, v1b.WINDOW)
    dead = v1.dead_flags(speed, ball, graded)
    long_gap = v1b.long_gap_flags(ball, graded)
    labels = v1b.build(vote, dead, use_hold=True, use_detector=True, long_gap=long_gap)
    labels[outofplay.flags(ball, tracking.PITCH, halves) & graded] = "D"
    return labels


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
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the CSV (name it <team_name>.csv)")
    args = parser.parse_args(argv)

    labels = make_labels()
    baseline.write_submission(args.out, labels)
    check_format(args.out)
    counts = {c: int((labels == c).sum()) for c in "HADX"}
    print("wrote {}".format(args.out))
    print("rows {}  H {}  A {}  D {}  X {}".format(len(labels), *[counts[c] for c in "HADX"]))
    print("sha256 {}".format(sha256(args.out)))
    print("validate with: python evaluation/validate_submission.py {}".format(args.out))


if __name__ == "__main__":
    main()
