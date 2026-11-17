"""Train and test the learned gap model (variant G) as pre-registered, apply it to our match, write
docs/results/gap_model.md and possession/gap_model.json.

    python possession/gapmodel_report.py

Needs the Metrica sample data in data/external/metrica_sample/ (see docs/REPRODUCE.md). Aggregates only.
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
import gapmodel  # noqa: E402
import gapscore  # noqa: E402
import tracking  # noqa: E402
import v1  # noqa: E402
import v1b  # noqa: E402
import missingness_summary as ms  # noqa: E402
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

OUT_DIR = os.path.join(HERE, "output", "v1e")
BOARD_DIR = os.path.join(HERE, "output", "board_v1e")
RESULT_MD = os.path.join(ROOT, "docs", "results", "gap_model.md")


def rule_e(rows):
    return np.array([r["length_s"] >= gapscore.LONG_S for r in rows], dtype=float)


def rule_scorecard_no_tag(rows):
    pts = np.array([2 * (r["length_s"] >= gapscore.LONG_S) + 2 * (np.isfinite(r["speed"]) and r["speed"] < gapscore.SLOW_MPS)
                    + r["edge_out"] + r["restart_in"] - 3 * r["keeper"] for r in rows])
    return (pts >= gapscore.THRESHOLD).astype(float)


def f1s(rows, pred):
    """(gap F1, frame-weighted F1) of 'dead'."""
    y = np.array([r["y"] for r in rows])
    tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    gap_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    nd = np.array([r["n_dead"] for r in rows])
    nf = np.array([r["n_frames"] for r in rows])
    tpf, fpf, fnf = (nd * pred).sum(), ((nf - nd) * pred).sum(), (nd * (1 - pred)).sum()
    return gap_f1, 2 * tpf / max(2 * tpf + fpf + fnf, 1)


def main():
    games = {g: gapmodel.sample_gaps(g) for g in (1, 2, 3)}
    data_rows = [{"game": g, "gaps": len(r), "dead gaps": int(sum(x["y"] for x in r)),
                  "dead share, gaps < 10 s": "{:.0%}".format(np.mean([x["y"] for x in r if x["length_s"] < 10])),
                  "dead share, gaps >= 10 s": "{:.0%}".format(np.mean([x["y"] for x in r if x["length_s"] >= 10])),
                  "keeper fires": int(sum(x["keeper"] for x in r))} for g, r in games.items()]

    # leave one game out
    cv = []
    for test in (1, 2, 3):
        train = [r for g in (1, 2, 3) if g != test for r in games[g]]
        model = gapmodel.fit(gapmodel.matrix(train), np.array([r["y"] for r in train]))
        rows = games[test]
        pred_g = (gapmodel.predict(model, gapmodel.matrix(rows)) >= 0.5).astype(float)
        for name, pred in (("G (learned)", pred_g), ("E rule (>= 10 s)", rule_e(rows)),
                           ("scorecard without tag", rule_scorecard_no_tag(rows))):
            gf, ff = f1s(rows, pred)
            cv.append({"test game": test, "method": name, "gap F1 (dead)": gf, "frame F1 (dead)": ff})
    cv = pd.DataFrame(cv)
    mean = cv.groupby("method")[["gap F1 (dead)", "frame F1 (dead)"]].mean().reset_index()
    g_mean = float(mean.loc[mean.method == "G (learned)", "frame F1 (dead)"].iloc[0])
    e_mean = float(mean.loc[mean.method == "E rule (>= 10 s)", "frame F1 (dead)"].iloc[0])
    ok_cv = g_mean >= e_mean + 0.02

    # final model on all three games
    all_rows = [r for g in (1, 2, 3) for r in games[g]]
    model = gapmodel.fit(gapmodel.matrix(all_rows), np.array([r["y"] for r in all_rows]))
    gapmodel.save(model)
    coef = pd.DataFrame({"feature": ["(intercept)"] + list(gapmodel.FEATURES),
                         "weight (per 1 sd)": [model["intercept"]] + model["coef"],
                         "train mean": [np.nan] + model["mean"], "train sd": [np.nan] + model["sd"]})

    # our match
    pos, ball, team, speed = tracking.load_tracking(with_speed=True)
    halves = [(a, b) for _, a, b in half_video_blocks(GAME)]
    graded = np.zeros(len(ball), dtype=bool)
    for a, b in halves:
        graded[a:b + 1] = True
    vote = v1b.votes(baseline.nearest_team(pos, ball, team), v1b.WINDOW)
    dead = v1.dead_flags(speed, ball, graded)
    labels_d = v1b.build(vote, dead, use_hold=True, use_detector=True)
    labels_e = v1b.build(vote, dead, use_hold=True, use_detector=True, long_gap=v1b.long_gap_flags(ball, graded))
    rows = gapmodel.gap_table(pos, ball, team, halves, tracking.PITCH)
    prob = gapmodel.predict(model, gapmodel.matrix(rows))
    labels_g = gapmodel.apply(labels_d, rows, prob)
    labels_g[v1.H2_PRE_KICKOFF[0]:v1.H2_PRE_KICKOFF[1] + 1] = "D"

    feat_cmp = pd.DataFrame({"feature": list(gapmodel.FEATURES),
                             "sample games mean": gapmodel.matrix(all_rows).mean(axis=0) if False else np.nanmean(gapmodel.matrix(all_rows), axis=0),
                             "our match mean": np.nanmean(gapmodel.matrix(rows), axis=0)})
    L = np.array([r["length_s"] for r in rows])
    pd_ = prob >= 0.5
    dec = pd.DataFrame([{"gaps": "< 10 s", "D": int((pd_ & (L < 10)).sum()), "live, keeper": int((~pd_ & (L < 10) & np.array([r["keeper_team"] is not None for r in rows])).sum()),
                         "live, hold": int((~pd_ & (L < 10) & np.array([r["keeper_team"] is None for r in rows])).sum())},
                        {"gaps": ">= 10 s", "D": int((pd_ & (L >= 10)).sum()), "live, keeper": int((~pd_ & (L >= 10) & np.array([r["keeper_team"] is not None for r in rows])).sum()),
                         "live, hold": int((~pd_ & (L >= 10) & np.array([r["keeper_team"] is None for r in rows])).sum())}])

    def stats(lab):
        count, mean_s = v1.possession_stats(lab)
        g = lab[graded]
        return {"possessions": count, "mean duration (s)": "{:.1f}".format(mean_s),
                **{c: "{:.1%}".format((g == c).mean()) for c in "HAD"}}
    structure = pd.DataFrame([{"variant": "E (v1c, current file)", **stats(labels_e)},
                              {"variant": "G (v1e, learned gap model)", **stats(labels_g)}])
    changed = (labels_e != labels_g) & graded
    trans = pd.Series(["{} -> {}".format(a, b) for a, b in zip(labels_e[changed], labels_g[changed])]).value_counts()
    trans_tab = trans.rename_axis("E -> G").rename("frames").reset_index()
    _, mean_g = v1.possession_stats(labels_g)
    share_d = float((labels_g[graded] == "D").mean())
    ok_plaus = (v1.PLAUSIBLE_POSSESSION_S[0] <= mean_g <= v1.PLAUSIBLE_POSSESSION_S[1]
                and v1.PLAUSIBLE_DEAD_SHARE[0] <= share_d <= v1.PLAUSIBLE_DEAD_SHARE[1])

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BOARD_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "possession_v1e.csv")
    baseline.write_submission(out_csv, labels_g)
    baseline.write_submission(os.path.join(BOARD_DIR, "E_v1c.csv"), labels_e)
    baseline.write_submission(os.path.join(BOARD_DIR, "G_gap_model.csv"), labels_g)
    board_out = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                                "--results", BOARD_DIR], capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
    start = next(i for i, l in enumerate(board_out) if l.startswith("scoreboard"))
    end = next((i for i, l in enumerate(board_out) if l.startswith("macroF1 would order")), len(board_out))
    board = "\n".join(board_out[start:end]).rstrip()
    macro = {p[1]: float(p[2]) for p in (l.split() for l in board_out[start:end]) if len(p) > 3 and p[1] in ("E_v1c", "G_gap_model")}
    ok_guard = macro["G_gap_model"] >= macro["E_v1c"] - 0.02
    validator = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out_csv],
                               capture_output=True, text=True, cwd=ROOT).stdout.strip()

    paths = asset_paths()
    prov = "\n".join([
        "- generated (UTC): {}".format(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python possession/gapmodel_report.py`",
        "- git commit: `{}`{}".format(ms.git("rev-parse", "HEAD"), " (tracked files had uncommitted changes)"
                                      if ms.git("status", "--porcelain", "--untracked-files=no") else ""),
    ] + ["- possession/{} sha256: `{}`".format(f, ms.sha256(os.path.join(HERE, f)))
         for f in ("gapmodel.py", "gapmodel_report.py", "gapscore.py", "gap_model.json", "v1b.py", "v1.py", "deadball.py")] + [
        "- training data: Metrica Sports open sample data, games 1 to 3 (github.com/metrica-sports/sample-data)",
        "- input tracking sha256: `{}`".format(ms.sha256(paths["atd_raw"])),
        "- G file sha256: `{}`".format(ms.sha256(out_csv)),
        "- python {} / numpy {} / pandas {}".format(sys.version.split()[0], np.__version__, pd.__version__),
    ])
    yes = lambda ok: "**pass**" if ok else "**FAIL**"
    fmt = lambda df: ms.pipe_table(df.round(3))
    md = """# Learned gap model (variant G, v1e)

Generated by `possession/gapmodel_report.py`. Pre-registered in `docs/worklog.md` ("2026-11-15 (learned gap model,
v1e)") before any model was fitted. Trained on other matches only (Metrica Sports open sample data, games 1 to 3).
Aggregates only.

## Provenance

{prov}

## 1. Training data (other matches)

A gap is dead when more than half its frames lie between the last event before a restart and the restart.

{data}

## 2. Held-out test: leave one game out (the real measurement)

Fit on two sample games, test on the third. F1 of "dead": per gap, and weighted by frames.

{cv}

Mean over the three test games:

{mean}

## 3. Final model (fitted on all three games)

Weights are per standard deviation of the feature (positive = more likely dead).

{coef}

## 4. Our match

Feature means, training games vs our match (a large difference means the model is used outside what it saw):

{featcmp}

Gap decisions:

{dec}

```
{validator}
```

{structure}

Graded frames relabelled from E to G ({nchanged} in total):

{trans}

## 5. Decision rule (written in advance)

1. mean held-out frame F1 of G >= E rule + 0.02: G {gm:.3f}, E {em:.3f} -> {ok_cv}
2. plausibility on our match (D 5% to 45%, mean possession 5 to 40 s): D {sd:.1%}, {mg:.1f} s -> {ok_plaus}
3. public-window macroF1 of G not below E by more than 0.02: G {mG:.4f}, E {mE:.4f} -> {ok_guard}
4. the team agrees: **open**

## 6. Official scorer, public 5 minutes (guard and description only; 17 possessions, cannot rank)

```
{board}
```

## 7. What this does not tell us

The held-out F1 is measured on other matches, whose tracking may lose the ball for other reasons than ours. The
full-match score on our match is unknown until the organisers publish results.
""".format(prov=prov, data=ms.pipe_table(pd.DataFrame(data_rows)), cv=fmt(cv), mean=fmt(mean), coef=fmt(coef),
           featcmp=fmt(feat_cmp), dec=ms.pipe_table(dec), validator=validator, structure=ms.pipe_table(structure),
           nchanged=int(changed.sum()), trans=ms.pipe_table(trans_tab), gm=g_mean, em=e_mean, ok_cv=yes(ok_cv),
           sd=share_d, mg=mean_g, ok_plaus=yes(ok_plaus), mG=macro["G_gap_model"], mE=macro["E_v1c"],
           ok_guard=yes(ok_guard), board=board)
    with open(RESULT_MD, "w") as handle:
        handle.write(md)
    print("wrote", os.path.relpath(RESULT_MD, ROOT))
    for df in (pd.DataFrame(data_rows), cv.round(3), mean.round(3), coef.round(3), feat_cmp.round(3), dec, structure, trans_tab):
        print(df.to_string(index=False))
        print()
    print("criteria: cv {} plausibility {} guard {}".format(ok_cv, ok_plaus, ok_guard))
    print(board)


if __name__ == "__main__":
    main()
