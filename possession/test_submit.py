"""Checks of the submission pipeline's building blocks on synthetic data. No match data needed.

    python possession/test_submit.py
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import submit  # noqa: E402


def test_controlled_codes_requires_one_team_in_range():
    pos = np.full((5, 4, 2), np.nan)
    team = np.array([0, 0, 1, 1])
    ball = np.array([[0.0, 0.0], [0.0, 0.0], [np.nan, np.nan], [0.0, 0.0], [0.0, 0.0]])
    pos[0, 0] = (1.0, 0.0)          # frame 0: only home in range -> 0
    pos[0, 2] = (5.0, 0.0)
    pos[1, 0] = (1.0, 0.0)          # frame 1: both in range -> contest -> -1
    pos[1, 2] = (1.0, 0.0)
    pos[2, 0] = (1.0, 0.0)          # frame 2: ball not tracked -> -1
    pos[2, 2] = (5.0, 0.0)
    pos[3, 0] = (3.0, 0.0)          # frame 3: nobody in range -> -1
    pos[3, 2] = (4.0, 0.0)
    pos[4, 0] = (10.0, 0.0)         # frame 4: only away in range -> 1
    pos[4, 2] = (0.5, 0.0)
    assert list(submit.controlled_codes(pos, ball, team, radius=1.5)) == [0, -1, -1, -1, 1]


def test_ball_in_flight_casts_no_vote():
    # 10 frames; the ball rests for 5 frames, then flies at 0.5 m/frame (12.5 m/s).
    ball = np.zeros((10, 2))
    ball[5:, 0] = 0.5 * np.arange(1, 6)
    speed = submit.ball_speed(ball)
    assert np.isnan(speed[:2]).all() and np.isnan(speed[-2:]).all()     # edges have no centred speed
    assert speed[2] == 0.0 and abs(speed[7] - 12.5) < 1e-9
    codes = np.zeros(10, dtype=np.int8)
    gated = submit.drop_in_flight(codes, ball, limit=7.0)
    # frame 4 spans the start of the flight at 6.25 m/s (still a vote); from frame 5 the ball is in flight
    assert list(gated) == [-1, -1, 0, 0, 0, -1, -1, -1, -1, -1]


def test_majority_ties_follow_the_surrounding_possession():
    codes = np.array([0, 1, 1, 0], dtype=np.int8)
    # frame 0 is a 1-1 tie between home and away; the next decisive window is away, so it is away.
    # frame 3 is a 1-1 tie; the previous decision is away, so it stays away. Neither goes to home.
    assert list(submit.majority(codes, window=1)) == [1, 1, 1, 1]


def test_majority_without_any_vote_is_minus_one():
    codes = np.full(10, -1, dtype=np.int8)
    assert (submit.majority(codes, window=3) == -1).all()


def test_hold_forward_fills_and_back_fills_the_head():
    segment = np.array([-1, -1, 1, -1, -1, 0, -1], dtype=np.int8)
    assert list(submit.hold(segment)) == [1, 1, 1, 1, 1, 0, 0]


def test_write_submission_is_lf_and_exact_header():
    labels = np.array(list("HADXH"), dtype="<U1")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "tiny.csv")
        submit.write_submission(path, labels)
        with open(path, "rb") as handle:
            raw = handle.read()
    assert raw.startswith(b"frame,label\n")
    assert b"\r" not in raw
    assert raw.count(b"\n") == 6                       # header + five rows, and nothing else


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print("{} checks passed".format(len(tests)))
