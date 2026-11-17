"""v1 submission: nearest-team evidence, a dead-ball detector, and a two-state Viterbi possession path.

    python possession/v1.py

Design and parameters were written in docs/worklog.md ("2026-11-12 (v1)") before this was run. No
parameter is chosen from any ground truth.

  1. Evidence       on a frame with the ball tracked, the team of the nearest tracked slot (0 home, 1 away).
  2. Dead ball      deadball.detect(theta 1.0 m/s, T 5 s) -> D; plus the second-half frames 110808-113741 (graded,
                    before kick-off), which FAQ.md states are all D.
  3. Possession     per half, a two-state Viterbi path. Being in the state that differs from the nearest team costs 1
                    per frame; frames without a tracked ball and frames flagged dead carry no evidence; a switch
                    costs LAMBDA frames of evidence (75 = 3 s, the challenge's own proxy for "controlled possession":
                    a player keeps the ball for a few seconds) and is free on dead frames. A missing ball in live play
                    therefore keeps the current state.
  4. Outside the two halves the label is X (never scored).

Writes possession/output/v1/possession_v1.csv and docs/results/possession_v1.md (provenance included).
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "explore"))

import baseline  # noqa: E402
import deadball  # noqa: E402
import tracking  # noqa: E402
import missingness_summary as ms  # noqa: E402  (provenance helpers)
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

THETA, MIN_SECONDS = 1.0, 5          # selected by the dead-ball detector's pre-registered procedure
LAMBDA = 75                          # switch penalty in frames (3 s at 25 fps)
SENSITIVITY = (25, 50, 75, 125, 200, 300)
H2_PRE_KICKOFF = (110808, 113741)    # graded frames before the second-half kick-off, all D (FAQ.md)
OUT_DIR = os.path.join(HERE, "output", "v1")
BOARD_DIR = os.path.join(HERE, "output", "board")
RESULT_MD = os.path.join(ROOT, "docs", "results", "possession_v1.md")
PLAUSIBLE_POSSESSION_S = (5, 40)
PLAUSIBLE_DEAD_SHARE = (0.05, 0.45)


def viterbi(codes, dead, penalty):
    """Best two-state path (0 home, 1 away) for one stretch of frames.

    codes   int8, 0/1 nearest team or -1 where the ball is not tracked
    dead    bool, True on frames flagged dead (no evidence there, switching is free)
    penalty cost of a switch, in frames of contrary evidence
    """
    n = len(codes)
    live = ~dead
    cost0 = ((codes == 1) & live).astype(np.float64)     # cost of being in state 0 on each frame
    cost1 = ((codes == 0) & live).astype(np.float64)
    switch = np.where(dead, 0.0, float(penalty))
    from_other = np.zeros((n, 2), dtype=bool)
    m0, m1 = cost0[0], cost1[0]
    for t in range(1, n):
        via1_to0, via0_to1 = m1 + switch[t], m0 + switch[t]
        if m0 <= via1_to0:
            n0 = m0
        else:
            n0, from_other[t, 0] = via1_to0, True
        if m1 <= via0_to1:
            n1 = m1
        else:
            n1, from_other[t, 1] = via0_to1, True
        m0, m1 = n0 + cost0[t], n1 + cost1[t]
    state = np.empty(n, dtype=np.int8)
    state[-1] = 0 if m0 <= m1 else 1
    for t in range(n - 1, 0, -1):
        state[t - 1] = (1 - state[t]) if from_other[t, state[t]] else state[t]
    return state


def assemble(codes, dead, penalty=LAMBDA):
    """Label array ('H', 'A', 'D', 'X') for every video frame."""
    labels = np.full(len(codes), "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        sl = slice(first, last + 1)
        path = viterbi(codes[sl], dead[sl], penalty)
        labels[sl] = np.where(path == 0, "H", "A")
        labels[sl][dead[sl]] = "D"
    return labels


def dead_flags(speed, ball, graded):
    """Dead-ball frames: the detector's stoppages plus the stated second-half pre-kick-off frames."""
    _, flag = deadball.detect(speed, ball, graded, THETA, MIN_SECONDS)
    flag[H2_PRE_KICKOFF[0]:H2_PRE_KICKOFF[1] + 1] = True
    return flag & graded


def possession_stats(labels):
    """(count, mean duration in s) of possessions: maximal runs of H or of A (D and X separate them)."""
    lengths = []
    for team in "HA":
        lengths += [e - s for s, e in deadball.runs(labels == team)]
    return len(lengths), (float(np.mean(lengths)) / deadball.FPS if lengths else float("nan"))


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout


def main():
    paths = asset_paths()
    print("loading ...")
    pos, ball, team, speed = tracking.load_tracking(with_speed=True)
    n = len(ball)
    graded = np.zeros(n, dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    codes = baseline.nearest_team(pos, ball, team)
    dead = dead_flags(speed, ball, graded)

    labels = assemble(codes, dead)
    os.makedirs(OUT_DIR, exist_ok=True)
    baseline.write_submission(os.path.join(OUT_DIR, "possession_v1.csv"), labels)

    count, mean_s = possession_stats(labels)
    g = labels[graded]
    share = {c: float((g == c).mean()) for c in "HAD"}
    ok_possession = PLAUSIBLE_POSSESSION_S[0] <= mean_s <= PLAUSIBLE_POSSESSION_S[1]
    ok_dead = PLAUSIBLE_DEAD_SHARE[0] <= share["D"] <= PLAUSIBLE_DEAD_SHARE[1]
    print("v1: H {:.1%}  A {:.1%}  D {:.1%} of graded frames; {} possessions of mean {:.1f} s".format(
        share["H"], share["A"], share["D"], count, mean_s))
    print("plausibility bounds: mean possession {} -> {}; dead share {} -> {}".format(
        PLAUSIBLE_POSSESSION_S, "ok" if ok_possession else "OUTSIDE", PLAUSIBLE_DEAD_SHARE,
        "ok" if ok_dead else "OUTSIDE"))

    # context: the reference variants over the same graded frames, and the switch-penalty sensitivity
    rows = []
    for name, window in baseline.REFERENCE_WINDOWS:
        ref = baseline.smooth_majority(codes, window)
        ref = np.where(graded, ref, "X")
        c, m = possession_stats(ref)
        rows.append({"submission": name, "possessions": c, "mean duration (s)": "{:.1f}".format(m),
                     "D share": "{:.1%}".format(float((ref[graded] == "D").mean()))})
    for lam in SENSITIVITY:
        lab = assemble(codes, dead, lam)
        c, m = possession_stats(lab)
        rows.append({"submission": "v1, switch penalty {} frames ({:.1f} s){}".format(
            lam, lam / 25.0, "  <- used" if lam == LAMBDA else ""), "possessions": c,
            "mean duration (s)": "{:.1f}".format(m), "D share": "{:.1%}".format(float((lab[graded] == "D").mean()))})
    table = pd.DataFrame(rows)

    validator = run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                     os.path.join(OUT_DIR, "possession_v1.csv")]).strip()

    # official scorer on the public window, v1 next to the reference variants (description only)
    os.makedirs(BOARD_DIR, exist_ok=True)
    for name, _ in baseline.REFERENCE_WINDOWS:
        src = os.path.join(baseline.OUT_DIR, name + ".csv")
        if os.path.exists(src):
            pd.read_csv(src).to_csv(os.path.join(BOARD_DIR, name + ".csv"), index=False)
    pd.read_csv(os.path.join(OUT_DIR, "possession_v1.csv")).to_csv(os.path.join(BOARD_DIR, "v1_penalty75.csv"), index=False)
    board_out = run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                     "--results", BOARD_DIR]).splitlines()
    start = next(i for i, l in enumerate(board_out) if l.startswith("scoreboard"))
    end = next((i for i, l in enumerate(board_out) if l.startswith("macroF1 would order")), len(board_out))
    board = "\n".join(board_out[start:end]).rstrip()

    prov = "\n".join([
        "- generated (UTC): {}".format(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python possession/v1.py`",
        "- git commit: `{}`{}".format(ms.git("rev-parse", "HEAD"),
                                      " (tracked files had uncommitted changes)"
                                      if ms.git("status", "--porcelain", "--untracked-files=no") else ""),
    ] + ["- possession/{} sha256: `{}`".format(f, ms.sha256(os.path.join(HERE, f)))
         for f in ("v1.py", "deadball.py", "baseline.py", "tracking.py")] + [
        "- input tracking sha256: `{}`".format(ms.sha256(paths["atd_raw"])),
        "- input tracking metadata sha256: `{}`".format(ms.sha256(paths["atd_meta"])),
        "- parameters: theta {} m/s, T {} s, switch penalty {} frames, second-half pre-kick-off frames {}-{}".format(
            THETA, MIN_SECONDS, LAMBDA, *H2_PRE_KICKOFF),
        "- python {} / numpy {} / pandas {}".format(sys.version.split()[0], np.__version__, pd.__version__),
    ])
    md = """# Possession v1: first full submission

Generated by `possession/v1.py`. Design and parameters were written in `docs/worklog.md`
("2026-11-12 (v1)") before this was run. Nothing here is chosen from any ground truth.

## Provenance

{prov}

## 1. The file

```
{validator}
```

Of the graded frames: H {h:.1%}, A {a:.1%}, D {d:.1%}. Possessions (runs of H or of A): {count}, mean {mean:.1f} s.
Plausibility bounds written in advance: mean possession {pb} s -> **{okp}**; dead share {db} -> **{okd}**.

## 2. Structure next to the reference baselines, and the switch penalty (a description)

Possessions are counted over graded frames only, for every row. The penalty values other than the used one
are shown so the sensitivity is visible. None of them is selected from anything.

{table}

## 3. Official scorer, public 5 minutes (correctness check and description only)

The public window has 17 possessions and 4 dead-ball stretches and cannot rank anything (`EVALUATION.md`,
`RULES.md`). No choice was made from it. v1 is `v1_penalty75`.

```
{board}
```

## 4. What this does not tell us

There is no full-match ground truth to check v1 against. The reference baseline with window 50 has a published
full-match spellF1 of 0.2433; v1's is unknown and may be lower or higher. The pieces v1 changes (dead ball from
players' speed, a Viterbi path, holding the label through an untracked ball) are reasoned, not measured, on the
full match. Long ball gaps are held on the assumption that possession does not change unseen; that has not
been tested.
""".format(prov=prov, validator=validator, h=share["H"], a=share["A"], d=share["D"], count=count, mean=mean_s,
           pb=PLAUSIBLE_POSSESSION_S, okp="inside" if ok_possession else "OUTSIDE",
           db=PLAUSIBLE_DEAD_SHARE, okd="inside" if ok_dead else "OUTSIDE",
           table=ms.pipe_table(table), board=board)
    with open(RESULT_MD, "w") as handle:
        handle.write(md)
    print("wrote", os.path.relpath(RESULT_MD, ROOT))
    print(table.to_string(index=False))
    print(board)


if __name__ == "__main__":
    main()
