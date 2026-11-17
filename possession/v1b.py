"""v1b: the reference baseline with two narrow changes, and the ablation that isolates them.

    python possession/v1b.py

Design written in docs/worklog.md ("2026-11-12 (v1b)") before this was run. The smoothing is the reference
one (centred majority vote over +/- 50 frames, ties to home); only two things change:

  hold      where the window holds no vote (ball not tracked), keep the previous vote instead of writing D
            (forward fill inside each half, back fill at the head of a half)
  detector  D on the frames the stoppage detector flags (theta 1.0 m/s, T 5 s) and on the second-half frames
            110808-113741 that FAQ.md states are all D, written over the vote

Five variants, reported with equal prominence:   A reference | B hold only | C detector only | D hold + detector (= v1b)
| E = D plus D on ball-untracked runs of 10 s or more (= v1c; pre-registered in worklog "gap check, v1c")

Writes possession/output/v1b/possession_v1b.csv (variant D) and docs/results/possession_v1b.md.
No parameter is chosen from any ground truth; the public window is shown as a description only.
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
import tracking  # noqa: E402
import v1  # noqa: E402  (dead_flags, possession_stats)
import missingness_summary as ms  # noqa: E402  (provenance helpers)
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

WINDOW = 50                                  # the reference's published-floor smoothing
OUT_DIR = os.path.join(HERE, "output", "v1b")
BOARD_DIR = os.path.join(HERE, "output", "board_v1b")
RESULT_MD = os.path.join(ROOT, "docs", "results", "possession_v1b.md")
VARIANTS = (("A", "reference, window 50", False, False),
            ("B", "hold only", True, False),
            ("C", "detector only", False, True),
            ("D", "hold + detector (v1b)", True, True),
            ("E", "D + long ball gaps (>=10 s) as D (v1c)", True, True))


def votes(codes, window):
    """Centred majority vote: 0 home, 1 away, -1 where the window holds no vote. Ties go to home."""
    home_c = np.concatenate(([0], np.cumsum(codes == 0)))
    away_c = np.concatenate(([0], np.cumsum(codes == 1)))
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - window), np.minimum(len(codes), index + window + 1)
    home, away = home_c[high] - home_c[low], away_c[high] - away_c[low]
    out = np.full(len(codes), -1, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], 0, 1)
    return out


def hold(segment):
    """Fill -1 with the previous known vote; at the head of the segment use the first known vote."""
    n = len(segment)
    last_known = np.maximum.accumulate(np.where(segment >= 0, np.arange(n), -1))
    filled = np.where(last_known >= 0, segment[np.maximum(last_known, 0)], -1)
    if (filled < 0).any() and (segment >= 0).any():
        filled[filled < 0] = segment[segment >= 0][0]
    return filled


LONG_GAP_FRAMES = 250                        # variant E: ball untracked for 10 s or more -> D (worklog "gap check, v1c")


def long_gap_flags(ball, graded, min_frames=LONG_GAP_FRAMES):
    """True on frames inside a run of untracked ball of at least min_frames, within a graded half."""
    miss = np.isnan(ball[:, 0])
    flags = np.zeros(len(ball), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        m = miss[first:last + 1].astype(int)
        d = np.diff(np.r_[0, m, 0])
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s >= min_frames:
                flags[first + s:first + e] = True
    return flags & graded


def build(vote, dead, use_hold, use_detector, long_gap=None):
    """Labels H/A/D/X for every video frame. long_gap: optional boolean array written as D (variant E)."""
    labels = np.full(len(vote), "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        seg = vote[first:last + 1]
        seg = hold(seg) if use_hold else seg
        labels[first:last + 1] = np.where(seg == 0, "H", np.where(seg == 1, "A", "D"))
    if use_detector:
        labels[dead] = "D"
    if long_gap is not None:
        labels[long_gap] = "D"
    return labels


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
    vote = votes(codes, WINDOW)
    dead = v1.dead_flags(speed, ball, graded)

    long_gap = long_gap_flags(ball, graded)
    made = {key: build(vote, dead, h, d, long_gap if key == "E" else None) for key, _, h, d in VARIANTS}

    # variant A must be the reference on every graded frame
    reference = pd.read_csv(os.path.join(baseline.REFERENCE_DIR, "baseline_smooth50.csv"))["label"].values
    a_differs = int((made["A"][graded] != reference[graded]).sum())
    print("variant A vs the committed reference window-50 labels, graded frames that differ: {}".format(a_differs))
    if a_differs:
        sys.exit("variant A does not reproduce the reference; stopping")

    os.makedirs(OUT_DIR, exist_ok=True)
    baseline.write_submission(os.path.join(OUT_DIR, "possession_v1b.csv"), made["D"])
    baseline.write_submission(os.path.join(OUT_DIR, "possession_v1c.csv"), made["E"])

    rows = []
    for key, name, _, _ in VARIANTS:
        lab = made[key]
        count, mean_s = v1.possession_stats(lab)
        g = lab[graded]
        rows.append({"variant": "{}: {}".format(key, name), "possessions": count,
                     "mean duration (s)": "{:.1f}".format(mean_s),
                     "H": "{:.1%}".format((g == "H").mean()), "A": "{:.1%}".format((g == "A").mean()),
                     "D": "{:.1%}".format((g == "D").mean())})
    structure = pd.DataFrame(rows)

    def changed(x, y):
        return int(((made[x] != made[y]) & graded).sum())
    a = made["A"]
    change_rows = [
        {"relabelled frames (graded)": "A -> B: D to H or A by hold", "frames": changed("A", "B")},
        {"relabelled frames (graded)": "A -> C: live to D by the detector", "frames": int(((a != made["C"]) & graded & (made["C"] == "D")).sum())},
        {"relabelled frames (graded)": "A -> D: all changes", "frames": changed("A", "D")},
        {"relabelled frames (graded)": "D -> E: live to D by long ball gaps", "frames": changed("D", "E")},
        {"relabelled frames (graded)": "A -> E: all changes", "frames": changed("A", "E")},
        {"relabelled frames (graded)": "graded frames in total", "frames": int(graded.sum())},
    ]
    changes = pd.DataFrame(change_rows)

    count, mean_s = v1.possession_stats(made["D"])
    share_d = float((made["D"][graded] == "D").mean())
    ok_p = v1.PLAUSIBLE_POSSESSION_S[0] <= mean_s <= v1.PLAUSIBLE_POSSESSION_S[1]
    ok_d = v1.PLAUSIBLE_DEAD_SHARE[0] <= share_d <= v1.PLAUSIBLE_DEAD_SHARE[1]
    validator = run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                     os.path.join(OUT_DIR, "possession_v1b.csv")]).strip()

    os.makedirs(BOARD_DIR, exist_ok=True)
    for key, name, _, _ in VARIANTS:
        short = {"A": "A_reference50", "B": "B_hold", "C": "C_detector", "D": "D_hold_detector", "E": "E_long_gap_D"}[key]
        pd.DataFrame({"frame": np.arange(n), "label": made[key]}).to_csv(os.path.join(BOARD_DIR, short + ".csv"), index=False)
    board_out = run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                     "--results", BOARD_DIR]).splitlines()
    start = next(i for i, l in enumerate(board_out) if l.startswith("scoreboard"))
    end = next((i for i, l in enumerate(board_out) if l.startswith("macroF1 would order")), len(board_out))
    board = "\n".join(board_out[start:end]).rstrip()

    prov = "\n".join([
        "- generated (UTC): {}".format(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python possession/v1b.py`",
        "- git commit: `{}`{}".format(ms.git("rev-parse", "HEAD"),
                                      " (tracked files had uncommitted changes)"
                                      if ms.git("status", "--porcelain", "--untracked-files=no") else ""),
    ] + ["- possession/{} sha256: `{}`".format(f, ms.sha256(os.path.join(HERE, f)))
         for f in ("v1b.py", "v1.py", "deadball.py", "baseline.py", "tracking.py")] + [
        "- input tracking sha256: `{}`".format(ms.sha256(paths["atd_raw"])),
        "- input tracking metadata sha256: `{}`".format(ms.sha256(paths["atd_meta"])),
        "- parameters: smoothing window +/-{} frames; detector theta {} m/s, T {} s; second-half pre-kick-off frames {}-{}".format(
            WINDOW, v1.THETA, v1.MIN_SECONDS, *v1.H2_PRE_KICKOFF),
        "- python {} / numpy {} / pandas {}".format(sys.version.split()[0], np.__version__, pd.__version__),
    ])
    md = """# Possession v1b: the reference baseline with two narrow changes

Generated by `possession/v1b.py`. Design written in `docs/worklog.md` ("2026-11-12 (v1b)") before this was run.
The smoothing is the reference's (window 50). Nothing here is chosen from any ground truth.

## Provenance

{prov}

## 1. Check: variant A is the reference

Variant A reproduces the committed `evaluation/results/baseline_smooth50.csv` on all {ng} graded frames
({ad} differ).

## 2. The file (variant D = v1b; variant E = v1c is written next to it)

```
{validator}
```

Plausibility bounds written in advance: mean possession {pb} s -> **{okp}** ({mean:.1f} s); dead share {db} ->
**{okd}** ({sd:.1%}).

## 3. The five variants (same smoothing; equal prominence)

{structure}

How many graded frames each change relabels:

{changes}

## 4. Official scorer, public 5 minutes (correctness check and description only)

The public window has 17 possessions and 4 dead-ball stretches and cannot rank anything (`EVALUATION.md`,
`RULES.md`). Nothing was chosen from it; all five variants are shown, not only the best looking.

```
{board}
```

## 5. What this does not tell us

There is no full-match ground truth. The reference (window 50) has a published full-match spellF1 of 0.2433;
the full-match score of v1b, and of B and C, is unknown. Holding assumes possession does not change while the
ball is untracked (untested). The detector covers long stoppages only (worklog finding 39), so a short stoppage
inside a missing-ball stretch is labelled as live play by the hold.
""".format(prov=prov, ng=int(graded.sum()), ad=a_differs, validator=validator, pb=v1.PLAUSIBLE_POSSESSION_S,
           okp="inside" if ok_p else "OUTSIDE", mean=mean_s, db=v1.PLAUSIBLE_DEAD_SHARE,
           okd="inside" if ok_d else "OUTSIDE", sd=share_d, structure=ms.pipe_table(structure),
           changes=ms.pipe_table(changes), board=board)
    with open(RESULT_MD, "w") as handle:
        handle.write(md)
    print("wrote", os.path.relpath(RESULT_MD, ROOT))
    print(structure.to_string(index=False))
    print(changes.to_string(index=False))
    print("plausibility: possession {} dead share {}".format("ok" if ok_p else "OUTSIDE", "ok" if ok_d else "OUTSIDE"))
    print(board)


if __name__ == "__main__":
    main()
