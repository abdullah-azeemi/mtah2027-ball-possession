"""Small checks of the pipeline's building blocks on synthetic data. No match data needed.

    python possession/test_pipeline.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import baseline  # noqa: E402
import deadball  # noqa: E402
import gapmodel  # noqa: E402
import gapscore  # noqa: E402
import outofplay  # noqa: E402
import v1  # noqa: E402
import v1b  # noqa: E402


def test_nearest_team_picks_closest_slot_and_handles_missing():
    pos = np.full((3, 4, 2), np.nan)
    pos[0, 0], pos[0, 1] = (0.0, 0.0), (10.0, 0.0)            # frame 0: slots 0 (home) and 1 (away)
    pos[1, 0], pos[1, 1] = (0.0, 0.0), (10.0, 0.0)
    pos[2, 0] = (0.0, 0.0)
    ball = np.array([[9.0, 0.0], [np.nan, np.nan], [1.0, 0.0]])
    team = np.array([0, 1, 0, 1])
    assert list(baseline.nearest_team(pos, ball, team)) == [1, -1, 0]      # away, no ball, home


def test_votes_tie_goes_to_home_and_empty_window_is_minus_one():
    codes = np.array([0, 1, -1, -1, -1, -1, -1, -1], dtype=np.int8)
    out = v1b.votes(codes, window=1)
    assert out[0] == 0 and out[1] == 0                         # one home vote, one away vote around frame 1: tie -> home
    assert out[7] == -1                                         # nothing within +/- 1 frame


def test_hold_forward_fills_and_back_fills_the_head():
    segment = np.array([-1, -1, 1, -1, -1, 0, -1], dtype=np.int8)
    assert list(v1b.hold(segment)) == [1, 1, 1, 1, 1, 0, 0]


def test_runs_and_stoppage_merging():
    mask = np.zeros(1000, dtype=bool)
    mask[100:300] = True
    mask[310:600] = True                                        # gap of 10 frames < 2 s: merged
    mask[800:810] = True                                        # too short for T = 5 s
    assert deadball.stoppages(mask, min_seconds=5) == [(100, 600)]


def test_possession_stats_counts_runs_of_one_team():
    labels = np.array(list("HHHDAAXXHH"))
    count, mean_s = v1.possession_stats(labels)
    assert count == 3 and abs(mean_s - (3 + 2 + 2) / 3 / 25.0) < 1e-9


def test_viterbi_needs_enough_evidence_to_switch():
    codes = np.array([0] * 100 + [1] * 10 + [0] * 100, dtype=np.int8)    # a 10-frame blip of the other team
    dead = np.zeros(len(codes), dtype=bool)
    assert (v1.viterbi(codes, dead, penalty=75) == 0).all()               # blip too short to justify two switches
    long_codes = np.array([0] * 100 + [1] * 300, dtype=np.int8)
    assert v1.viterbi(long_codes, np.zeros(400, dtype=bool), penalty=75)[-1] == 1


PITCH = (105.3, 68.0)


def test_gapscore_restart_spots_and_edges():
    assert gapscore.restart_spot(np.array([1.0, 1.0]), PITCH)            # corner
    assert gapscore.restart_spot(np.array([52.0, 34.5]), PITCH)          # centre spot
    assert gapscore.restart_spot(np.array([3.0, 30.0]), PITCH)           # goal area
    assert gapscore.restart_spot(np.array([40.0, 67.5]), PITCH)          # on a touchline
    assert not gapscore.restart_spot(np.array([40.0, 30.0]), PITCH)      # open play
    assert gapscore.edge_out(np.array([104.0, 30.0]), PITCH) and not gapscore.edge_out(np.array([50.0, 30.0]), PITCH)


def _synthetic_match(ball_before, ball_after, keeper_xy, gap=(100, 399), n=600):
    """Home slots near x = 0 (home defends end 0), away near the other end; ball missing on `gap`."""
    pos = np.full((n, 4, 2), np.nan)
    pos[:, 0] = keeper_xy                    # home's deepest slot
    pos[:, 1] = (30.0, 34.0)                 # home outfield
    pos[:, 2] = (70.0, 34.0)                 # away outfield
    pos[:, 3] = (100.0, 34.0)                # away keeper
    team = np.array([0, 0, 1, 1])
    ball = np.tile(np.array([50.0, 34.0]), (n, 1))
    ball[gap[0] - 1], ball[gap[1] + 1] = ball_before, ball_after
    ball[gap[0]:gap[1] + 1] = np.nan
    return pos, ball, team


def test_gapscore_keeper_gap_goes_to_keeper_team_and_long_gap_to_d():
    pos, ball, team = _synthetic_match((10.0, 30.0), (12.0, 35.0), keeper_xy=(11.0, 35.0))
    speed = np.full(len(ball), 2.0)                                       # players moving: not slow
    rows = gapscore.gap_features(pos, ball, team, speed, np.array([], dtype=np.int64), [(0, 599)], PITCH)
    assert len(rows) == 1 and rows[0]["keeper"] and rows[0]["keeper_team"] == 0 and rows[0]["decision"] == "keeper"
    labels = gapscore.apply(np.array(["A"] * 600), rows)
    assert (labels[100:400] == "H").all() and labels[99] == "A"
    # same gap but the ball reappears in open play: long (2 points) -> D
    pos, ball, team = _synthetic_match((10.0, 30.0), (40.0, 30.0), keeper_xy=(11.0, 35.0))
    rows = gapscore.gap_features(pos, ball, team, speed, np.array([], dtype=np.int64), [(0, 599)], PITCH)
    assert not rows[0]["keeper"] and rows[0]["points"] == 2 and rows[0]["decision"] == "D"
    # a set-piece tag inside the gap switches the keeper rule off
    pos, ball, team = _synthetic_match((10.0, 30.0), (12.0, 35.0), keeper_xy=(11.0, 35.0))
    rows = gapscore.gap_features(pos, ball, team, speed, np.array([350]), [(0, 599)], PITCH)
    assert rows[0]["tag"] and not rows[0]["keeper"] and rows[0]["decision"] == "D"


def test_outofplay_dead_from_out_until_fast_restart():
    n = 1000
    ball = np.tile(np.array([50.0, 34.0]), (n, 1))
    ball[100:120] = (60.0, 70.0)                                          # 2 m beyond the touchline for 0.8 s
    ball[120:300] = (60.0, 67.0)                                          # carried back, still
    ball[300:400] = (60.0, 67.0) + np.outer(np.arange(100), (0.0, -0.4))  # thrown in at 10 m/s
    ball[400:] = ball[399]
    st = outofplay.stretches(ball, PITCH, [(0, n - 1)])
    assert len(st) == 1 and st[0][0] == 100 and 295 <= st[0][1] <= 302
    ball2 = np.tile(np.array([50.0, 34.0]), (n, 1))
    ball2[100:105] = (60.0, 70.0)                                         # out for only 5 frames: noise, ignored
    assert outofplay.stretches(ball2, PITCH, [(0, n - 1)]) == []


def test_gapmodel_logistic_regression_learns_a_separable_feature():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(gapmodel.FEATURES)))
    y = (X[:, 0] > 0).astype(float)                                       # dead iff the first feature is positive
    model = gapmodel.fit(X, y)
    assert model["coef"][0] > 1 and abs(model["coef"][1]) < 0.5
    assert ((gapmodel.predict(model, X) >= 0.5) == (y == 1)).mean() > 0.95


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print("{} checks passed".format(len(tests)))
