"""Evaluate the dead-ball detector against Smart Tagging restarts, and describe it on the public window.

    python possession/deadball_report.py

Follows the procedure written in docs/worklog.md ("2026-11-12 (dead ball)") BEFORE it was run:

  * evidence: the 59 Smart Tagging `SET PIECES` tags (automatic, not human labels). Long restarts are
    corner, free kick and kick-off (23 tags); short restarts are throw-in and goal kick (36 tags);
  * R (recall proxy): share of long-restart tags with a detected stoppage that ends in
    [tag start - 3 s, tag start + 10 s]; also given for throw-ins and goal kicks;
  * P (precision proxy): share of detected stoppages that end in that window around any set-piece tag.
    The tags are incomplete, so P is pessimistic;
  * selection: the grid cell maximising the harmonic mean of R (long restarts) and P; ties go to the
    larger T, then the lower theta. The public ground truth plays no part in the selection;
  * failure criterion: if no cell has R >= 0.5 and P >= 0.3 the rule is declared not usable.

The public window (11185-18684) is described afterwards for the chosen cell only. It is not a score
to improve and no choice is made from it.

Writes docs/results/deadball_detector.md with a provenance block.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "explore"))

import deadball  # noqa: E402
import tracking  # noqa: E402
import missingness_summary as ms  # noqa: E402  (provenance helpers)
from games import asset_paths, half_video_blocks, GAME  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULT_MD = os.path.join(ROOT, "docs", "results", "deadball_detector.md")
THETAS = (0.5, 0.6, 0.8, 1.0)
MIN_SECONDS = (3, 5, 8)
WINDOW = (-3 * deadball.FPS, 10 * deadball.FPS)   # accepted stoppage end, relative to the tag start
LONG = ("Corner Kick", "Free Kick", "Kick Off (After Goal)", "Kick Off (Start)")
SHORT = ("Throw-in", "Goal Kick")


def restart_tags():
    """[(type, start_frame)] of the set-piece tags."""
    return [(e["tags"].get("Type"), e["start_frame"]) for e in load_smart_tagging.load()
            if (e["code"] or "").strip().lower() == "set pieces"]


def ends_near(stops, start):
    return any(start + WINDOW[0] <= end <= start + WINDOW[1] for _, end in stops)


def score_cell(stops, tags, flag, graded):
    """R for long and short restarts, P, dead share, median stoppage length."""
    long_tags = [s for t, s in tags if t in LONG]
    short_tags = [s for t, s in tags if t in SHORT]
    all_starts = [s for _, s in tags]
    r_long = np.mean([ends_near(stops, s) for s in long_tags])
    r_short = np.mean([ends_near(stops, s) for s in short_tags])
    precision = (np.mean([any(s + WINDOW[0] <= e <= s + WINDOW[1] for s in all_starts) for _, e in stops])
                 if stops else np.nan)
    f1 = (2 * r_long * precision / (r_long + precision)) if stops and (r_long + precision) > 0 else 0.0
    return {"R long": r_long, "R short": r_short, "P": precision, "F1": f1, "stoppages": len(stops),
            "dead share": flag[graded].mean(),
            "median length (s)": np.median([(e - s) / deadball.FPS for s, e in stops]) if stops else np.nan}


def describe_public(flag):
    """How the chosen rule lines up with the public ground truth. Description only."""
    gt = pd.read_csv(GT_PATH)["label"].values
    sl = slice(11185, 18685)
    g, f = gt[sl], flag[sl]
    dead_gt = g == "D"
    rows = [{"measure": "ground-truth D frames", "value": int(dead_gt.sum())},
            {"measure": "frames the rule calls dead", "value": int(f.sum())},
            {"measure": "dead frames that are ground-truth D (precision)",
             "value": "{:.1f}%".format(100 * dead_gt[f].mean()) if f.any() else "-"},
            {"measure": "ground-truth D frames the rule finds (recall)",
             "value": "{:.1f}%".format(100 * f[dead_gt].mean())}]
    # per ground-truth dead-ball stretch
    runs, current = [], None
    for i, label in enumerate(g):
        if label != current:
            runs.append([label, i, i])
            current = label
        else:
            runs[-1][2] = i
    stretches = [{"ground-truth D stretch": "{}-{}".format(11185 + s, 11185 + e),
                  "length": "{:.1f} s".format((e - s + 1) / 25.0),
                  "rule calls dead on": "{:.0f}%".format(100 * f[s:e + 1].mean())}
                 for label, s, e in runs if label == "D"]
    false_pos = []
    for start, end in deadball.runs(f & ~dead_gt):
        if end - start >= 25:
            values, counts = np.unique(g[start:end], return_counts=True)
            false_pos.append({"frames the rule calls dead but are not D": "{}-{}".format(11185 + start, 11185 + end - 1),
                              "length": "{:.1f} s".format((end - start) / 25.0),
                              "ground truth there": ", ".join("{} {}".format(v, c) for v, c in zip(values, counts))})
    return rows, stretches, false_pos


def main():
    paths = asset_paths()
    print("loading ...")
    pos, ball, team, speed = tracking.load_tracking(with_speed=True)
    n = len(ball)
    graded = np.zeros(n, dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    tags = restart_tags()
    print("set-piece tags: {} (long {}, short {})".format(
        len(tags), sum(t in LONG for t, _ in tags), sum(t in SHORT for t, _ in tags)))

    rows, cache = [], {}
    for theta in THETAS:
        for minimum in MIN_SECONDS:
            stops, flag = deadball.detect(speed, ball, graded, theta, minimum)
            cache[(theta, minimum)] = (stops, flag)
            rows.append(dict({"theta (m/s)": theta, "T (s)": minimum}, **score_cell(stops, tags, flag, graded)))
    grid = pd.DataFrame(rows)

    usable = grid[(grid["R long"] >= 0.5) & (grid["P"] >= 0.3)]
    best = grid.sort_values(["F1", "T (s)", "theta (m/s)"], ascending=[False, False, True]).iloc[0]
    chosen = (best["theta (m/s)"], int(best["T (s)"]))
    verdict = ("USABLE under the failure criterion" if len(usable) else
               "NOT USABLE under the failure criterion (no cell has R >= 0.5 and P >= 0.3)")
    print(verdict)
    print("selected cell: theta {} m/s, T {} s".format(*chosen))

    shown = grid.copy()
    for col in ("R long", "R short", "P", "F1", "dead share"):
        shown[col] = shown[col].map(lambda v: "{:.2f}".format(v) if np.isfinite(v) else "-")
    shown["median length (s)"] = shown["median length (s)"].map(lambda v: "{:.1f}".format(v) if np.isfinite(v) else "-")
    shown["selected"] = ["<-" if (r["theta (m/s)"], r["T (s)"]) == chosen else "" for _, r in grid.iterrows()]

    stops, flag = cache[chosen]
    summary, stretches, false_pos = describe_public(flag)

    type_rows = []
    for typ in LONG + SHORT:
        starts = [s for t, s in tags if t == typ]
        type_rows.append({"restart type": typ, "tags": len(starts),
                          "stoppage ends within [-3 s, +10 s]": "{:.0f}%".format(
                              100 * np.mean([ends_near(stops, s) for s in starts]))})

    prov = "\n".join([
        "- generated (UTC): {}".format(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python possession/deadball_report.py`",
        "- git commit: `{}`{}".format(ms.git("rev-parse", "HEAD"),
                                      " (tracked files had uncommitted changes)"
                                      if ms.git("status", "--porcelain", "--untracked-files=no") else ""),
        "- possession/deadball.py sha256: `{}`".format(ms.sha256(os.path.join(HERE, "deadball.py"))),
        "- possession/deadball_report.py sha256: `{}`".format(ms.sha256(os.path.abspath(__file__))),
        "- possession/tracking.py sha256: `{}`".format(ms.sha256(os.path.join(HERE, "tracking.py"))),
        "- input tracking sha256: `{}`".format(ms.sha256(paths["atd_raw"])),
        "- input tracking metadata sha256: `{}`".format(ms.sha256(paths["atd_meta"])),
        "- input Smart Tagging sha256: `{}`".format(ms.sha256(paths["smart_tagging"])),
        "- public ground truth sha256 (description only): `{}`".format(ms.sha256(GT_PATH)),
        "- python {} / numpy {} / pandas {}".format(sys.version.split()[0], np.__version__, pd.__version__),
    ])
    md = """# Dead-ball detector: evaluation against Smart Tagging restarts

Generated by `possession/deadball_report.py`. The procedure, selection rule and failure criterion were
written in `docs/worklog.md` ("2026-11-12 (dead ball)") before this was run. Rule: `possession/deadball.py`.

## Provenance

{prov}

## 1. Grid (all cells)

R long = share of the {nl} corner / free-kick / kick-off tags with a detected stoppage ending in
[tag start - 3 s, tag start + 10 s]; R short = the same for the {ns} throw-in / goal-kick tags;
P = share of detected stoppages ending in that window around any set-piece tag (pessimistic, the tags
are incomplete); F1 = harmonic mean of R long and P.

{grid}

**Verdict:** {verdict}. Selected cell by the pre-registered rule: theta {theta} m/s, T {minimum} s.

## 2. The selected rule, by restart type

{types}

## 3. The selected rule on the public window (description, not a score)

The public ground truth plays no part in the selection above. Here it only describes the chosen rule.
The window has 4 dead-ball stretches, so these are leads, not results.

{summary}

{stretches}

{false_pos}
""".format(prov=prov, nl=sum(t in LONG for t, _ in tags), ns=sum(t in SHORT for t, _ in tags),
           grid=ms.pipe_table(shown), verdict=verdict, theta=chosen[0], minimum=chosen[1],
           types=ms.pipe_table(pd.DataFrame(type_rows)), summary=ms.pipe_table(pd.DataFrame(summary)),
           stretches=ms.pipe_table(pd.DataFrame(stretches)),
           false_pos=("Stretches of 1 s or more that the rule calls dead and the ground truth does not:\n\n"
                      + ms.pipe_table(pd.DataFrame(false_pos))) if false_pos else
           "No stretch of 1 s or more is called dead by the rule outside the ground-truth D frames.")
    os.makedirs(os.path.dirname(RESULT_MD), exist_ok=True)
    with open(RESULT_MD, "w") as handle:
        handle.write(md)
    print("wrote", os.path.relpath(RESULT_MD, ROOT))
    print(shown.to_string(index=False))


if __name__ == "__main__":
    main()
