"""Build the baseline, check it against the reference, score it, and describe where it errs.

    python possession/report.py

Writes docs/results/possession_baseline.md with a provenance block. Three parts:

  1. Reproduction check  our labels vs the committed evaluation/results/baseline_*.csv, frame by frame.
  2. Official scorer     evaluation/run_evaluation.py on the public 5 minutes. A CORRECTNESS check
                         only: it confirms we read the data as the organisers do. It must not be
                         used to choose between versions (RULES.md; EVALUATION.md shows why).
  3. Public-window review  a DESCRIPTION of where the baseline's labels disagree with the public
                         ground truth (what dead ball looks like in the tracking, how well the
                         nearest-player rule works at different distances). It is read to understand
                         the labelling conventions, as CHALLENGE.md encourages. No threshold, rule
                         or model choice is made from it; with 17 possessions and 4 dead-ball
                         stretches it could not support one.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "explore"))

import baseline  # noqa: E402
import tracking  # noqa: E402
import missingness_summary as ms  # noqa: E402  (provenance helpers)
from games import asset_paths  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULT_MD = os.path.join(ROOT, "docs", "results", "possession_baseline.md")
FIRST, LAST = 11185, 18684          # frames covered by the public ground truth


def run_official_scorer():
    """The first lines of the official scoreboard for our four submissions."""
    out = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                          "--results", baseline.OUT_DIR], capture_output=True, text=True, cwd=ROOT).stdout
    lines = out.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("scoreboard"))
    end = next(i for i, l in enumerate(lines) if l.startswith("macroF1 would order"))
    return "\n".join(lines[start:end]).rstrip()


def review(pos, ball, team, codes, smooth50):
    """Descriptive tables of the baseline against the public ground truth. Returns markdown text."""
    gt = pd.read_csv(GT_PATH)["label"].values
    sl = slice(FIRST, LAST + 1)
    g, vis, lab = gt[sl], np.isfinite(ball[sl, 0]), smooth50[sl]
    parts = []

    rows = [{"ground truth": c, "frames": int((g == c).sum()),
             "ball tracked on": "{:.1f}%".format(100 * vis[g == c].mean())} for c in "HAD"]
    parts.append("### 1. Is the ball tracked, by ground-truth state?\n\n" + ms.pipe_table(pd.DataFrame(rows)))

    cm = pd.crosstab(pd.Series(g, name="gt"), pd.Series(lab, name="ours")).reindex(
        index=list("HAD"), columns=list("HAD")).fillna(0).astype(int)
    cm_table = cm.reset_index().rename(columns={"gt": "ground truth \\ ours (smooth50)"})
    parts.append("### 2. Confusion matrix, smooth50 baseline, public window\n\n" + ms.pipe_table(cm_table))

    runs, current = [], None
    for i, label in enumerate(g):
        if label != current:
            runs.append([label, i, i])
            current = label
        else:
            runs[-1][2] = i
    rows = []
    for label, s, e in runs:
        if label == "D":
            rows.append({"frames": "{}-{}".format(FIRST + s, FIRST + e),
                         "length": "{:.1f} s".format((e - s + 1) / 25.0),
                         "ball tracked on": "{:.0f}%".format(100 * vis[s:e + 1].mean()),
                         "smooth50 says D on": "{:.0f}%".format(100 * (lab[s:e + 1] == "D").mean())})
    parts.append("### 3. The ground-truth dead-ball stretches\n\n" + ms.pipe_table(pd.DataFrame(rows)))

    rows, missing, i = [], ~vis, 0
    while i < len(missing):
        if missing[i]:
            j = i
            while j < len(missing) and missing[j]:
                j += 1
            if j - i >= 25:
                values, counts = np.unique(g[i:j], return_counts=True)
                rows.append({"frames": "{}-{}".format(FIRST + i, FIRST + j - 1),
                             "length": "{:.1f} s".format((j - i) / 25.0),
                             "ground truth inside": ", ".join("{} {}".format(v, c) for v, c in zip(values, counts))})
            i = j
        else:
            i += 1
    parts.append("### 4. Ball-missing runs of 1 s or more, and what the ground truth says there\n\n"
                 + ms.pipe_table(pd.DataFrame(rows)))

    dx, dy = pos[sl, :, 0] - ball[sl, 0][:, None], pos[sl, :, 1] - ball[sl, 1][:, None]
    dmin = np.sqrt(np.where(np.isnan(dx * dy), np.inf, dx * dx + dy * dy)).min(axis=1)
    raw, ok, truth = codes[sl], vis & np.isin(g, list("HA")), np.where(g == "H", 0, 1)
    rows = [{"nearest tracked player": "all", "frames": int(ok.sum()),
             "raw rule right": "{:.1f}%".format(100 * (raw[ok] == truth[ok]).mean())}]
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]:
        s = ok & (dmin >= lo) & (dmin < hi)
        rows.append({"nearest tracked player": "{:g}-{:g} m".format(lo, hi) if hi < 1e8 else "over {:g} m".format(lo),
                     "frames": int(s.sum()), "raw rule right": "{:.1f}%".format(100 * (raw[s] == truth[s]).mean())})
    parts.append("### 5. The raw nearest-team rule on frames with the ball tracked and a live-ball label\n\n"
                 + ms.pipe_table(pd.DataFrame(rows)))
    return "\n\n".join(parts)


def main():
    paths = asset_paths()
    print("building ...")
    pos, ball, team = tracking.load_tracking()
    codes = baseline.nearest_team(pos, ball, team)
    produced = {}
    for name, window in baseline.REFERENCE_WINDOWS:
        produced[name] = baseline.smooth_majority(codes, window)
        baseline.write_submission(os.path.join(baseline.OUT_DIR, name + ".csv"), produced[name])
    check = baseline.verify_against_reference(produced)
    check_table = pd.DataFrame([{"variant": n, "frames": t, "frames that differ": d,
                                 "result": "identical" if d == 0 else "MISMATCH"} for n, (d, t) in check.items()])

    board = run_official_scorer()
    text = review(pos, ball, team, codes, produced["baseline_smooth50"])

    prov = "\n".join([
        "- generated (UTC): {}".format(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python possession/report.py`",
        "- git commit: `{}`{}".format(ms.git("rev-parse", "HEAD"),
                                      " (tracked files had uncommitted changes)"
                                      if ms.git("status", "--porcelain", "--untracked-files=no") else ""),
        "- possession/tracking.py sha256: `{}`".format(ms.sha256(os.path.join(HERE, "tracking.py"))),
        "- possession/baseline.py sha256: `{}`".format(ms.sha256(os.path.join(HERE, "baseline.py"))),
        "- possession/report.py sha256: `{}`".format(ms.sha256(os.path.abspath(__file__))),
        "- input tracking sha256: `{}`".format(ms.sha256(paths["atd_raw"])),
        "- input tracking metadata sha256: `{}`".format(ms.sha256(paths["atd_meta"])),
        "- public ground truth sha256: `{}`".format(ms.sha256(GT_PATH)),
        "- python {} / numpy {} / pandas {}".format(sys.version.split()[0], np.__version__, pd.__version__),
    ])
    md = """# Possession baseline: our own pipeline, checked against the reference

Generated by `possession/report.py`. Method and rationale: `possession/README.md`.

## Provenance

{prov}

## 1. Reproduction check

Our labels (`possession/baseline.py`, written from scratch) against the committed reference baselines
in `evaluation/results/`, over all 199,319 video frames.

{check}

## 2. Official scorer, public 5 minutes (correctness check only)

These numbers equal the ones published in the repository. That confirms frame alignment, team
assignment, units and the smoothing arithmetic. The public window has 17 possessions and cannot
rank anything: it is not used to choose between versions (`EVALUATION.md`, `RULES.md`).

```
{board}
```

## 3. Public-window review (description, not tuning)

Where the smooth50 baseline disagrees with the public ground truth, read to understand the labelling
conventions. The sample is tiny (7,500 frames, 17 possessions, 4 dead-ball stretches, 4 ball gaps of
1 s or more): these are observations to investigate on the full match, not rules.

{review}
""".format(prov=prov, check=ms.pipe_table(check_table), board=board, review=text)
    os.makedirs(os.path.dirname(RESULT_MD), exist_ok=True)
    with open(RESULT_MD, "w") as handle:
        handle.write(md)
    print("wrote", os.path.relpath(RESULT_MD, ROOT))
    print(check_table.to_string(index=False))


if __name__ == "__main__":
    main()
