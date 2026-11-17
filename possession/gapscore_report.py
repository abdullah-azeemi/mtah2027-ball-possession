"""Run the pre-registered gap scorecard (variant F) and write docs/results/gap_scorecard.md.

    python possession/gapscore_report.py

Aggregates only: counts, frame numbers, video times, booleans and points. No coordinates, no tracking values.
Writes possession/output/v1d/possession_v1d.csv (git-ignored) and scores E and F on the public window.
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
import deadball_report  # noqa: E402
import gapscore  # noqa: E402
import tracking  # noqa: E402
import v1  # noqa: E402
import v1b  # noqa: E402
import missingness_summary as ms  # noqa: E402
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

OUT_DIR = os.path.join(HERE, "output", "v1d")
BOARD_DIR = os.path.join(HERE, "output", "board_v1d")
RESULT_MD = os.path.join(ROOT, "docs", "results", "gap_scorecard.md")
FEATURES = ("long", "slow", "edge_out", "restart_in", "tag", "keeper")


def make_variants():
    """(labels E, labels F, gap rows, graded mask)."""
    pos, ball, team, speed = tracking.load_tracking(with_speed=True)
    halves = [(a, b) for _, a, b in half_video_blocks(GAME)]
    graded = np.zeros(len(ball), dtype=bool)
    for a, b in halves:
        graded[a:b + 1] = True
    vote = v1b.votes(baseline.nearest_team(pos, ball, team), v1b.WINDOW)
    dead = v1.dead_flags(speed, ball, graded)
    labels_d = v1b.build(vote, dead, use_hold=True, use_detector=True)
    labels_e = v1b.build(vote, dead, use_hold=True, use_detector=True, long_gap=v1b.long_gap_flags(ball, graded))
    rows = gapscore.gap_features(pos, ball, team, gapscore.rolling_speed(speed),
                                 gapscore.tag_start_frames(deadball_report.restart_tags()), halves, tracking.PITCH)
    labels_f = gapscore.apply(labels_d, rows)
    labels_f[v1.H2_PRE_KICKOFF[0]:v1.H2_PRE_KICKOFF[1] + 1] = "D"   # FAQ.md: all D, unchanged
    return labels_e, labels_f, rows, graded


def mmss(f):
    return "%d:%02d" % divmod(int(f / gapscore.FPS), 60)


def main():
    labels_e, labels_f, rows, graded = make_variants()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BOARD_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "possession_v1d.csv")
    baseline.write_submission(out_csv, labels_f)
    baseline.write_submission(os.path.join(BOARD_DIR, "E_v1c.csv"), labels_e)
    baseline.write_submission(os.path.join(BOARD_DIR, "F_gap_scorecard.csv"), labels_f)

    gaps = pd.DataFrame(rows)
    n_keeper = int(gaps["keeper"].sum())
    by_feature = pd.DataFrame([{"feature": k, "points": gapscore.POINTS[k], "gaps": int(gaps[k].sum()),
                                "of which short (< 10 s)": int((gaps[k] & ~gaps["long"]).sum())} for k in FEATURES])
    by_decision = (gaps.assign(group=np.where(gaps["long"], ">= 10 s", "< 10 s"))
                   .pivot_table(index="decision", columns="group", values="s", aggfunc="count", fill_value=0)
                   .reset_index())
    by_points = gaps.groupby("points").size().rename("gaps").reset_index()

    def stats(lab):
        count, mean_s = v1.possession_stats(lab)
        g = lab[graded]
        return {"possessions": count, "mean duration (s)": "{:.1f}".format(mean_s),
                **{c: "{:.1%}".format((g == c).mean()) for c in "HAD"}}
    structure = pd.DataFrame([{"variant": "E (v1c, current file)", **stats(labels_e)},
                              {"variant": "F (v1d, gap scorecard)", **stats(labels_f)}])
    changed = (labels_e != labels_f) & graded
    trans = pd.Series(["{} -> {}".format(a, b) for a, b in zip(labels_e[changed], labels_f[changed])]).value_counts()
    trans_tab = trans.rename_axis("E -> F").rename("frames").reset_index()

    per_gap = gaps.copy()
    per_gap["video time"] = [mmss(s) + " to " + mmss(e) for s, e in zip(gaps["s"], gaps["e"])]
    per_gap["E writes"] = np.where(gaps["long"], "D", "kept")
    per_gap["F writes"] = ["keeper " + "HA"[r["keeper_team"]] if r["decision"] == "keeper"
                           else ("D" if r["decision"] == "D" else "kept") for r in rows]
    per_gap["length (s)"] = gaps["length_s"].map("{:.1f}".format)
    for k in FEATURES:
        per_gap[k] = gaps[k].map({True: "x", False: ""})
    per_gap = per_gap[["s", "e", "video time", "length (s)", *FEATURES, "points", "E writes", "F writes"]]
    per_gap = per_gap.rename(columns={"s": "start frame", "e": "end frame"})
    differs = per_gap[per_gap["E writes"] != per_gap["F writes"]]

    _, mean_f = v1.possession_stats(labels_f)
    share_d = float((labels_f[graded] == "D").mean())
    ok_keeper = 1 <= n_keeper <= 15
    ok_plaus = (v1.PLAUSIBLE_POSSESSION_S[0] <= mean_f <= v1.PLAUSIBLE_POSSESSION_S[1]
                and v1.PLAUSIBLE_DEAD_SHARE[0] <= share_d <= v1.PLAUSIBLE_DEAD_SHARE[1])

    board_out = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                                "--results", BOARD_DIR], capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
    start = next(i for i, l in enumerate(board_out) if l.startswith("scoreboard"))
    end = next((i for i, l in enumerate(board_out) if l.startswith("macroF1 would order")), len(board_out))
    board = "\n".join(board_out[start:end]).rstrip()
    macro = {}
    for line in board_out[start:end]:
        parts = line.split()
        if len(parts) > 3 and parts[1] in ("E_v1c", "F_gap_scorecard"):
            macro[parts[1]] = float(parts[2])
    ok_guard = macro["F_gap_scorecard"] >= macro["E_v1c"] - 0.02
    validator = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out_csv],
                               capture_output=True, text=True, cwd=ROOT).stdout.strip()

    paths = asset_paths()
    prov = "\n".join([
        "- generated (UTC): {}".format(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python possession/gapscore_report.py`",
        "- git commit: `{}`{}".format(ms.git("rev-parse", "HEAD"),
                                      " (tracked files had uncommitted changes)"
                                      if ms.git("status", "--porcelain", "--untracked-files=no") else ""),
    ] + ["- possession/{} sha256: `{}`".format(f, ms.sha256(os.path.join(HERE, f)))
         for f in ("gapscore.py", "gapscore_report.py", "v1b.py", "v1.py", "deadball.py", "baseline.py", "tracking.py")] + [
        "- input tracking sha256: `{}`".format(ms.sha256(paths["atd_raw"])),
        "- F file sha256: `{}`".format(ms.sha256(out_csv)),
        "- python {} / numpy {} / pandas {}".format(sys.version.split()[0], np.__version__, pd.__version__),
    ])
    yes = lambda ok: "**pass**" if ok else "**FAIL**"
    md = """# Gap scorecard (variant F, v1d)

Generated by `possession/gapscore_report.py`. Rule pre-registered in `docs/worklog.md` ("2026-11-15 (gap scorecard,
v1d)") before any feature was computed. Aggregates only: no coordinates or tracking values.

## Provenance

{prov}

## 1. Gaps and features

{ngaps} ball-missing gaps inside the graded halves. Points: 2 long + 2 slow + 1 edge_out + 1 restart_in + 2 tag
- 3 keeper; 2 or more means `D`; a keeper gap is given to the keeper's team.

{by_feature}

Decisions by gap length:

{by_decision}

Gaps by points:

{by_points}

## 2. The file

```
{validator}
```

{structure}

Graded frames relabelled from E to F ({nchanged} in total):

{trans}

## 3. Decision rule (written in advance)

1. `keeper` fires on 1 to 15 gaps: {n_keeper} gaps -> {ok_keeper}
2. plausibility (D share 5% to 45%, mean possession 5 to 40 s): D {share_d:.1%}, {mean_f:.1f} s -> {ok_plaus}
3. public-window macroF1 of F not below E by more than 0.02: F {mf:.4f}, E {me:.4f} -> {ok_guard}
4. the team reads the per-gap table and agrees: **open**

## 4. Official scorer, public 5 minutes (guard and description only; 17 possessions, cannot rank)

```
{board}
```

## 5. Gaps where F differs from E

{differs}

## 6. All gaps

{per_gap}

## 7. What this does not tell us

There is no full-match ground truth, so whether F beats E on the full match is unknown. The keeper rule assumes the
deepest tracked slot is the keeper and that he still holds the ball when it is tracked again. Short-gap evidence is weak.
""".format(prov=prov, ngaps=len(gaps), by_feature=ms.pipe_table(by_feature), by_decision=ms.pipe_table(by_decision),
           by_points=ms.pipe_table(by_points), validator=validator, structure=ms.pipe_table(structure),
           nchanged=int(changed.sum()), trans=ms.pipe_table(trans_tab), n_keeper=n_keeper, ok_keeper=yes(ok_keeper),
           share_d=share_d, mean_f=mean_f, ok_plaus=yes(ok_plaus), mf=macro["F_gap_scorecard"], me=macro["E_v1c"],
           ok_guard=yes(ok_guard), board=board, differs=ms.pipe_table(differs) if len(differs) else "(none)",
           per_gap=ms.pipe_table(per_gap))
    with open(RESULT_MD, "w") as handle:
        handle.write(md)
    print("wrote", os.path.relpath(RESULT_MD, ROOT))
    print(by_feature.to_string(index=False))
    print(by_decision.to_string(index=False))
    print(structure.to_string(index=False))
    print(trans_tab.to_string(index=False))
    print("keeper gaps {} -> {}; plausibility -> {}; guard -> {}".format(n_keeper, ok_keeper, ok_plaus, ok_guard))
    print(differs.to_string(index=False))
    print(board)


if __name__ == "__main__":
    main()
