"""Data for ball recovery: tracking -> 5 Hz arrays, time-ordered splits, masking, evaluation trials.

The task ("ball recovery") is: given the players' positions over a window and the ball where it is
tracked, estimate the ball where it is not tracked. There is no hand-made label anywhere: the only
supervision is the automatically tracked ball itself. We hide stretches of ball that ARE tracked,
ask a method to reconstruct them, and score the reconstruction against what was hidden.

Why the design looks the way it does (see docs/worklog.md, "Design constraints"):

* Everything is cut in **time order**. Frames 1/25 s apart are near duplicates, so a random split
  of frames or windows would leak. Training, validation and test are separate stretches of the
  match with a buffer between them, and no window crosses a half or a split boundary.
* Hidden stretches are **contiguous** and as long as real gaps (seconds), not single frames.
* Evaluation uses **fixed trials** (same hidden stretches for every method), built with a seed, with
  tracked ball on both sides of each hidden stretch, because real gaps have that too.

Coordinates stay in metres here (x along the pitch 0..105.3, y across 0..68). Normalisation happens
in the model. Time step = 5 frames = 0.2 s (5 Hz); a window is 150 steps = 30 s.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "explore"))

from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
import frame_explorer as fe  # noqa: E402  (slot -> team mapping only)

FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
N_SLOTS = 46
STRIDE = 5                      # frames per step -> 5 Hz
HZ = FPS // STRIDE
WINDOW = 30 * HZ                # 150 steps = 30 s
CONTEXT = 2 * HZ                # tracked-ball context required on each side of an evaluation gap
MINUTE = 60 * HZ                # steps per minute


# Loading -----------------------------------------------------------------------------------------
def load_halves(data_dir=None):
    """{half: {"pos": (T,46,2) m, "ball": (T,2) m, "frame0": first video frame}} at 5 Hz, NaN = missing.

    Step i of a half is video frame frame0 + STRIDE * i. Only graded frames (the two halves) are
    used. Also returns the (46,) team array: 0 = Barcelona (home), 1 = Athletic (away).
    """
    paths = asset_paths(data_dir=data_dir)
    slot_team, side_of_team = fe.MatchData._read_teams(paths["atd_meta"])
    team = np.array([0 if side_of_team[slot_team[s]] == "H" else 1 for s in range(1, N_SLOTS + 1)],
                    dtype=np.int64)

    blocks = {half: (first, last) for half, first, last in half_video_blocks(GAME)}
    wanted = {}
    for half, (first, last) in blocks.items():
        wanted[half] = np.arange(first, last + 1, STRIDE)
    arrays = {half: {"pos": np.full((len(f), N_SLOTS, 2), np.nan, dtype=np.float32),
                     "ball": np.full((len(f), 2), np.nan, dtype=np.float32),
                     "frame0": int(f[0])} for half, f in wanted.items()}
    index_of = {}
    for half, frames in wanted.items():
        for i, frame in enumerate(frames):
            index_of[int(frame)] = (half, i)

    with open(paths["atd_raw"], "r") as handle:
        for frame, line in enumerate(handle):
            if frame not in index_of:
                continue
            half, i = index_of[frame]
            _, slots, ball = line.rstrip("\n").split(":")
            tokens = slots.split(";")
            for k, token in enumerate(tokens):
                if not token.startswith("NaN"):
                    x, y, _ = token.split(",")
                    arrays[half]["pos"][i, k] = (float(x) * PITCH_L, float(y) * PITCH_W)
            if not ball.startswith("NaN"):
                bx, by, _ = ball.split(",")
                arrays[half]["ball"][i] = (float(bx) * PITCH_L, float(by) * PITCH_W)
    return arrays, team


# Time-ordered split -----------------------------------------------------------------------------
# Minutes are counted from the first graded frame of the half. The second half is graded from
# frame 110808 but kicks off at 113742, so its first ~2 minutes are dead ball before kick-off.
# Split "v2" (2026-11-12). Version 1 was: train = all of half 1 + half 2 minutes 0-20, val = half 2
# minutes 21-31, test = half 2 minutes 32-end. That test period was scored once for the first model
# (docs/results/ball_recovery_bilstm.md) and therefore cannot count as untouched evidence for any
# model chosen afterwards. v2 keeps it as "test_seen" (reported, flagged) and takes the last part
# of half 1 out of training as a new, clean "test" that no decision has ever looked at.
SPLIT_MINUTES = {
    # split: [(half, first_minute, last_minute_exclusive or None for "to the end")]
    "train": [(1, 0, 36), (2, 0, 20)],
    # 1 minute buffers (half 1: 36..38, half 2: 20..21) are unused, so periods never touch
    "val": [(2, 21, 31)],          # used only to choose the epoch
    "test_seen": [(2, 32, None)],  # already looked at once: NOT clean evidence
    "test": [(1, 38, None)],       # clean: never used for any decision
}
SPLIT_VERSION = "v2"


def split_segments(arrays):
    """{split: [(half, start_step, end_step)]} following SPLIT_MINUTES; end is exclusive."""
    out = {}
    for split, parts in SPLIT_MINUTES.items():
        out[split] = []
        for half, m0, m1 in parts:
            length = len(arrays[half]["ball"])
            start = m0 * MINUTE
            end = length if m1 is None else min(length, m1 * MINUTE)
            if end - start >= WINDOW:
                out[split].append((half, start, end))
    return out


# Training windows and masks ---------------------------------------------------------------------
def sample_windows(arrays, segments, rng, batch, min_observed=0.4, flip=True):
    """A batch of random windows from the given segments (time order is kept inside a window).

    Returns pos (B, WINDOW, 46, 2) and ball (B, WINDOW, 2) in metres, NaN = missing. Windows with
    less than ``min_observed`` of the ball tracked are redrawn: they carry too little supervision.
    ``flip`` mirrors the pitch horizontally and/or vertically at random (the paper's augmentation;
    it also makes first- and second-half data look alike, since the teams swap ends).
    """
    lengths = np.array([e - s - WINDOW + 1 for _, s, e in segments], dtype=np.float64)
    prob = lengths / lengths.sum()
    pos_out, ball_out = [], []
    while len(pos_out) < batch:
        seg = segments[rng.choice(len(segments), p=prob)]
        half, s, e = seg
        start = s + int(rng.integers(0, e - s - WINDOW + 1))
        ball = arrays[half]["ball"][start:start + WINDOW]
        if np.isfinite(ball[:, 0]).mean() < min_observed:
            continue
        pos = arrays[half]["pos"][start:start + WINDOW]
        pos, ball = pos.copy(), ball.copy()
        if flip:
            if rng.random() < 0.5:
                pos[..., 0] = PITCH_L - pos[..., 0]
                ball[:, 0] = PITCH_L - ball[:, 0]
            if rng.random() < 0.5:
                pos[..., 1] = PITCH_W - pos[..., 1]
                ball[:, 1] = PITCH_W - ball[:, 1]
        pos_out.append(pos)
        ball_out.append(ball)
    return np.stack(pos_out), np.stack(ball_out)


def training_mask(ball, rng, p_hide_all=0.25, gap_seconds=(1.0, 25.0), max_gaps=3):
    """Which ball steps are hidden from the model's INPUT, shape (B, WINDOW) bool (True = hidden).

    Always hidden: steps where the ball is really missing. In addition, with probability
    ``p_hide_all`` the whole ball is hidden (the model must work from the players alone, as in the
    paper's imputation experiment); otherwise 1 to ``max_gaps`` contiguous stretches are hidden,
    lengths log-uniform in ``gap_seconds`` (real gaps are seconds long: worklog finding 16).
    """
    hidden = ~np.isfinite(ball[..., 0])
    lo, hi = np.log(gap_seconds[0] * HZ), np.log(gap_seconds[1] * HZ)
    for b in range(len(ball)):
        if rng.random() < p_hide_all:
            hidden[b] = True
            continue
        for _ in range(int(rng.integers(1, max_gaps + 1))):
            length = int(np.exp(rng.uniform(lo, hi)))
            start = int(rng.integers(0, WINDOW - 1))
            hidden[b, start:start + length] = True
    return hidden


# Evaluation trials ------------------------------------------------------------------------------
def make_trials(arrays, segments, rng, gap_seconds=(1, 2, 5, 10)):
    """Fixed evaluation trials: a hidden stretch of known length inside tracked-ball context.

    For each segment and each gap length, stretches are placed left to right on parts of the
    timeline where the ball is tracked for the whole stretch AND for CONTEXT steps on both sides,
    spaced so that no two trials of one length overlap. Each trial also carries a 30 s window
    (inside the segment) that contains it, positioned at random but fixed by the seed.

    Returns a list of dicts: half, win_start, gap_start, gap_end (both relative to the window start,
    end exclusive), seconds.
    """
    trials = []
    for half, s, e in segments:
        ball_ok = np.isfinite(arrays[half]["ball"][s:e, 0])
        for seconds in gap_seconds:
            length = int(seconds * HZ)
            need = length + 2 * CONTEXT
            # prefix sums of "ball missing" give the count of missing steps in any interval in O(1)
            miss = np.concatenate([[0], np.cumsum(~ball_ok)])
            position = 0
            while position + need <= e - s:
                if miss[position + need] - miss[position] == 0:
                    rel_gap_start = position + CONTEXT
                    # window: random placement that keeps the gap and its context inside it
                    lo = max(0, position + need - WINDOW)
                    hi = min(position, e - s - WINDOW)
                    if hi < lo:
                        position += 1
                        continue
                    win = int(rng.integers(lo, hi + 1))
                    trials.append({"half": half, "win_start": s + win,
                                   "gap_start": rel_gap_start - win,
                                   "gap_end": rel_gap_start - win + length, "seconds": seconds})
                    position += need          # next trial of this length starts after this one
                else:
                    position += 1
    return trials


def trial_arrays(arrays, trials):
    """Stack the windows of the trials: pos (N, WINDOW, 46, 2), ball truth (N, WINDOW, 2)."""
    pos = np.stack([arrays[t["half"]]["pos"][t["win_start"]:t["win_start"] + WINDOW] for t in trials])
    ball = np.stack([arrays[t["half"]]["ball"][t["win_start"]:t["win_start"] + WINDOW] for t in trials])
    return pos, ball


def trial_input_ball(ball_truth, trials):
    """The ball the method is allowed to see: truth with each trial's gap (and real gaps) hidden."""
    shown = ball_truth.copy()
    for i, t in enumerate(trials):
        shown[i, t["gap_start"]:t["gap_end"]] = np.nan
    return shown
