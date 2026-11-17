"""Generate the submission, check it, score it, and write docs/results/submission.md.

    python possession/submit_report.py

The submission itself is one command:

    python possession/submit.py --out <team_name>.csv

This report adds the evidence needed for the technical document: the validator output, the official
scorer on the public 5 minutes, the per-class view and the tracking-only structure of the full match.
The public window is a correctness check and a description, not a tuning set (RULES.md,
EVALUATION.md); with 17 possessions it cannot rank two submissions reliably.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "explore"))

import submit  # noqa: E402
import tracking  # noqa: E402
import v1  # noqa: E402
import missingness_summary as ms  # noqa: E402  (provenance helpers)
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

OUT_DIR = os.path.join(HERE, "output", "submit")
BOARD_DIR = os.path.join(HERE, "output", "board_submit")
RESULT_MD = os.path.join(ROOT, "docs", "results", "submission.md")
FIRST, LAST = 11185, 18684          # frames covered by the public ground truth
REFERENCE = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout


def scoreboard():
    os.makedirs(BOARD_DIR, exist_ok=True)
    shutil.copyfile(os.path.join(OUT_DIR, "submit.csv"), os.path.join(BOARD_DIR, "submit.csv"))
    if os.path.exists(REFERENCE):
        shutil.copyfile(REFERENCE, os.path.join(BOARD_DIR, "baseline_smooth50.csv"))
    lines = run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                 "--results", BOARD_DIR]).splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("scoreboard"))
    end = next((i for i, l in enumerate(lines) if l.startswith("macroF1 would order")), len(lines))
    kept = [l for l in lines[start:end] if not l.startswith("Wrote ")]      # no local paths in the report
    return "\n".join(kept).rstrip()


def public_view(labels):
    """Per-class F1 and confusion on the public window, computed with the organisers' module."""
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
    sl = slice(FIRST, LAST + 1)
    result = pm.evaluate(pm.label_codes(gt[sl]), pm.label_codes(labels[sl]))
    rows = [{"label": k, "precision": "{:.3f}".format(v["precision"]),
             "recall": "{:.3f}".format(v["recall"]), "F1": "{:.3f}".format(v["f1"]),
             "support": v["support"]} for k, v in result["per_class"].items()]
    return result, pd.DataFrame(rows)


def main():
    paths = asset_paths()
    print("loading tracking ...")
    pos, ball, team, speed = tracking.load_tracking(with_speed=True)
    labels = submit.make_labels(pos, ball, team, speed)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "submit.csv")
    submit.write_submission(out, labels)

    validator = run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out]).strip()
    board = scoreboard()
    result, per_class = public_view(labels)

    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    count, mean_s = v1.possession_stats(labels)
    shares = {c: float((labels[graded] == c).mean()) for c in "HADX"}

    # possession lengths, computed directly so the table is transparent
    rows = []
    for team_code in "HA":
        lengths = [e - s for s, e in _runs(labels == team_code)]
        rows.append({"team": team_code, "possessions": len(lengths),
                     "mean duration (s)": "{:.1f}".format(np.mean(lengths) / 25.0),
                     "median duration (s)": "{:.1f}".format(np.median(lengths) / 25.0)})
    structure = pd.DataFrame(rows)

    prov = "\n".join([
        "- generated (UTC): {}".format(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python possession/submit_report.py`",
        "- git commit: `{}`{}".format(ms.git("rev-parse", "HEAD"),
                                      " (tracked files had uncommitted changes)"
                                      if ms.git("status", "--porcelain", "--untracked-files=no") else ""),
    ] + ["- possession/{} sha256: `{}`".format(f, ms.sha256(os.path.join(HERE, f)))
         for f in ("submit.py", "tracking.py", "v1.py", "v1b.py", "outofplay.py", "deadball.py", "baseline.py")] + [
        "- input tracking sha256: `{}`".format(ms.sha256(paths["atd_raw"])),
        "- input tracking metadata sha256: `{}`".format(ms.sha256(paths["atd_meta"])),
        "- parameters: control radius {} m, vote window +/-{} frames, in-flight gate {} m/s".format(
            submit.CONTROL_RADIUS_M, submit.VOTE_WINDOW, submit.FLIGHT_SPEED_MS),
        "- python {} / numpy {} / pandas {}".format(sys.version.split()[0], np.__version__, pd.__version__),
    ])

    md = """# Possession with controlled-ball evidence

Generated by `possession/submit_report.py`. Method and rationale: the docstring of
`possession/submit.py`; one-command generator: `python possession/submit.py --out <team_name>.csv`.

## Provenance

{prov}

## 1. The file

```
{validator}
```

Graded-frame shares: H {h:.1%}, A {a:.1%}, D {d:.1%}. Possessions (maximal H or A runs over the whole
match): {count}, mean {mean:.1f} s.

{structure}

## 2. Official scorer, public 5 minutes (correctness check and description only)

{board}

Per-class on the public window:

{per_class}

## 3. What changed against the adopted T_E candidate

| | T_E | submitted |
|---|---|---|
| per-frame evidence | team of the nearest tracked slot, any distance | team of the nearest slot, only when **exactly one team** has a player within {radius:g} m of the ball |
| contested / loose ball | counted as evidence | no vote; the last possession is kept (`hold`) |
| ball in flight (faster than {flight:g} m/s) | counted as evidence | no vote (a pass or clearance is not control) |
| vote window | +/-50 frames | +/-{window} frames (gating removes about half the votes) |
| vote tie | home | previous decided possession |
| dead ball | T_E rules | unchanged |
| outside the halves | X | unchanged |

The change is the challenge's own rule ("In phases where the ball is contested or loose, possession
stays with the last team that had it") applied to the frame-level evidence. Widening the vote window
compensates for the frames gating removes, so the number of contributing frames per decision is
about the same as the reference's.

## 4. What this does not tell us

There is no full-match ground truth. The public window has 17 possessions, so its spellF1 has a
wide uncertainty and the gain shown here may not hold over 158,857 graded frames. The dead-ball
rules are unchanged, so the dead-ball share of the full match is exactly T_E's. The main risk of
gating is a live phase in which the tracking loses the ball's nearest player for several seconds:
the method then keeps the old team, which is the intended behaviour for a loose ball but wrong if
control quietly changed hands.
""".format(prov=prov, validator=validator, board=board,
           per_class=ms.pipe_table(per_class), structure=ms.pipe_table(structure),
           h=shares["H"], a=shares["A"], d=shares["D"], count=count, mean=mean_s,
           radius=submit.CONTROL_RADIUS_M, window=submit.VOTE_WINDOW,
           flight=submit.FLIGHT_SPEED_MS)

    with open(RESULT_MD, "w") as handle:
        handle.write(md)
    print("wrote", os.path.relpath(RESULT_MD, ROOT))
    print("public: macroF1 {:.4f}  spellF1 {:.4f}  possessions {}".format(
        result["macro_f1"], result["spell_f1_mean"], result["n_pred_spells"]))
    print("sha256", submit.sha256(out))


def _runs(mask):
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    change = np.diff(padded)
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


if __name__ == "__main__":
    main()
