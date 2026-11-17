"""Our own end-to-end possession baseline: whoever is nearest the ball has it, smoothed.

This is deliberately the same method as the reference baseline in starter/baseline.py, written from
scratch in this repository's own structure. The point of doing it first:

  1. a pipeline that goes tracking -> per-frame label -> CSV -> validator -> scorer, which every later
     idea plugs into;
  2. a CHECK that we read the data the same way the organisers do: the labels produced here must be
     identical, frame for frame, to the committed evaluation/results/baseline_*.csv. If they are,
     frame alignment, team assignment, units and the smoothing arithmetic are right.

Method (no model, no training, no labels of any kind)
  step 1  per frame with the ball tracked: the team of the tracked slot nearest to the ball.
          Where the ball is not tracked there is no answer (code -1).
  step 2  centred majority vote over +/- `window` frames; D where the window holds no answer.

Run
    python possession/baseline.py                       # the four reference windows, then the check
    python possession/baseline.py --windows 50 --no-check
    python evaluation/run_evaluation.py --results possession/output/submissions   # official scorer

Scoring on the public 5 minutes is a CORRECTNESS check only (RULES.md): it must not be used to choose
between versions.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import tracking  # noqa: E402

OUT_DIR = os.path.join(HERE, "output", "submissions")
REFERENCE_DIR = os.path.join(ROOT, "evaluation", "results")
# Half-widths in frames of the four published variants (25 frames = 1 s).
REFERENCE_WINDOWS = (("baseline_raw", 0), ("baseline_smooth25", 25),
                     ("baseline_smooth50", 50), ("baseline_smooth125", 125))
NOBODY = -1


# Step 1 ------------------------------------------------------------------------------------------
def nearest_team(pos, ball, team, chunk=20000):
    """Team code (0 home, 1 away) of the slot nearest the ball per frame; -1 if there is no answer.

    No answer means: the ball is not tracked, or no slot has a position. Distances are Euclidean in
    metres (both axes scaled, the pitch is not square). Ties go to the lowest slot index.
    """
    out = np.full(len(ball), NOBODY, dtype=np.int8)
    for start in range(0, len(ball), chunk):
        sl = slice(start, start + chunk)
        dx = pos[sl, :, 0] - ball[sl, 0][:, None]
        dy = pos[sl, :, 1] - ball[sl, 1][:, None]
        d2 = dx * dx + dy * dy                                  # NaN wherever ball or slot is missing
        d2 = np.where(np.isnan(d2), np.inf, d2)
        nearest = d2.argmin(axis=1)
        found = np.isfinite(d2.min(axis=1))
        out[sl] = np.where(found, team[nearest], NOBODY)
    return out


# Step 2 ------------------------------------------------------------------------------------------
def smooth_majority(codes, window):
    """Labels 'H'/'A'/'D' from team codes: centred majority vote over +/- `window` frames.

    window 0 means no smoothing. D wherever the window holds no answer. Ties go to the home side.
    Cumulative sums make the cost independent of the window.
    """
    if window <= 0:
        return np.where(codes == 0, "H", np.where(codes == 1, "A", "D"))
    home_c = np.concatenate(([0], np.cumsum(codes == 0)))
    away_c = np.concatenate(([0], np.cumsum(codes == 1)))
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - window), np.minimum(len(codes), index + window + 1)
    home, away = home_c[high] - home_c[low], away_c[high] - away_c[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def write_submission(path, labels):
    """The submission format: header `frame,label`, one row per video frame."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)


# Check against the reference ----------------------------------------------------------------------
def verify_against_reference(produced, reference_dir=REFERENCE_DIR):
    """{name: (n_frames_that_differ, n_frames)} comparing our labels with the committed baselines."""
    report = {}
    for name, labels in produced.items():
        reference = pd.read_csv(os.path.join(reference_dir, name + ".csv"))["label"].values
        report[name] = (int((reference != labels).sum()), len(labels))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--windows", type=int, nargs="+", help="half-widths to write (default: the four reference ones)")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--no-check", action="store_true", help="skip the comparison with the committed baselines")
    args = parser.parse_args(argv)

    print("loading tracking ...")
    pos, ball, team = tracking.load_tracking()
    print("  ball tracked on {:.1f}% of all frames; slots: {} home, {} away".format(
        100 * np.isfinite(ball[:, 0]).mean(), int((team == 0).sum()), int((team == 1).sum())))
    codes = nearest_team(pos, ball, team)

    variants = ([("baseline_smooth{}".format(w) if w else "baseline_raw", w) for w in args.windows]
                if args.windows else list(REFERENCE_WINDOWS))
    produced = {}
    for name, window in variants:
        labels = smooth_majority(codes, window)
        write_submission(os.path.join(args.out, name + ".csv"), labels)
        produced[name] = labels
        print("  {:<20} window +/-{:<4} H {:>6}  A {:>6}  D {:>6}".format(
            name, window, *[int((labels == c).sum()) for c in "HAD"]))

    if not args.no_check:
        print("\ncheck against the committed reference baselines (frames that differ):")
        report = verify_against_reference({n: l for n, l in produced.items()
                                           if os.path.exists(os.path.join(REFERENCE_DIR, n + ".csv"))})
        for name, (differ, total) in report.items():
            print("  {:<20} {:>6} of {} differ   {}".format(name, differ, total,
                                                           "IDENTICAL" if differ == 0 else "MISMATCH"))
        if any(d for d, _ in report.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
