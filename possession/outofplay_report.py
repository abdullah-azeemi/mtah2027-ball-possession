"""Run the pre-registered out-of-play detector (T) and write docs/results/out_of_play.md. Aggregates only.

    python possession/outofplay_report.py

Needs possession/gap_model.json (for T_G) and, for section 1, the Metrica sample data in data/external/metrica_sample/.
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
import outofplay  # noqa: E402
import tracking  # noqa: E402
import v1  # noqa: E402
import v1b  # noqa: E402
import missingness_summary as ms  # noqa: E402
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

BOARD_DIR = os.path.join(HERE, "output", "board_t")
RESULT_MD = os.path.join(ROOT, "docs", "results", "out_of_play.md")
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def sample_rows():
    rows, n_str, tp, fp = [], 0, 0, 0
    for g in (1, 2, 3):
        pos, team, ball, dead, halves = gapmodel.load_sample_csv(g) if g < 3 else gapmodel.load_sample_epts()
        play = np.zeros(len(ball), dtype=bool)
        for a, b in halves:
            play[a:b + 1] = True
        st = outofplay.stretches(ball, gapmodel.SAMPLE_PITCH, halves)
        f = outofplay.flags(ball, gapmodel.SAMPLE_PITCH, halves) & play
        tracked = ~np.isnan(ball[:, 0])
        t_dead = play & dead & tracked
        rows.append({"game": g, "stretches": len(st), "frames flagged": int(f.sum()),
                     "precision (frames)": "{:.2f}".format(dead[f].mean()) if f.any() else "-",
                     "tracked dead frames": int(t_dead.sum()),
                     "of which found": "{:.1%}".format((f & t_dead).sum() / max(t_dead.sum(), 1))})
        n_str += len(st)
        tp += int((f & dead).sum())
        fp += int((f & ~dead).sum())
    return rows, n_str, (tp / (tp + fp) if tp + fp else float("nan"))


def main():
    srows, n_str, prec = sample_rows()
    measurable = n_str >= 20
    ok1 = measurable and prec >= 0.6

    pos, ball, team, speed = tracking.load_tracking(with_speed=True)
    halves = [(a, b) for _, a, b in half_video_blocks(GAME)]
    graded = np.zeros(len(ball), dtype=bool)
    for a, b in halves:
        graded[a:b + 1] = True
    st = outofplay.stretches(ball, tracking.PITCH, halves)
    t = outofplay.flags(ball, tracking.PITCH, halves) & graded
    lens = np.array([(e - s + 1) / outofplay.FPS for s, e in st])

    vote = v1b.votes(baseline.nearest_team(pos, ball, team), v1b.WINDOW)
    dead = v1.dead_flags(speed, ball, graded)
    labels_d = v1b.build(vote, dead, use_hold=True, use_detector=True)
    labels = {"E": v1b.build(vote, dead, use_hold=True, use_detector=True, long_gap=v1b.long_gap_flags(ball, graded))}
    rows = gapmodel.gap_table(pos, ball, team, halves, tracking.PITCH)
    labels["G"] = gapmodel.apply(labels_d, rows, gapmodel.predict(gapmodel.load(), gapmodel.matrix(rows)))
    labels["G"][v1.H2_PRE_KICKOFF[0]:v1.H2_PRE_KICKOFF[1] + 1] = "D"
    for base in ("E", "G"):
        lab = labels[base].copy()
        lab[t] = "D"
        labels["T_" + base] = lab

    def stats(lab):
        count, mean_s = v1.possession_stats(lab)
        g = lab[graded]
        return count, mean_s, {c: (g == c).mean() for c in "HAD"}
    struct = []
    plaus = {}
    for k in ("E", "G", "T_E", "T_G"):
        count, mean_s, share = stats(labels[k])
        plaus[k] = (v1.PLAUSIBLE_POSSESSION_S[0] <= mean_s <= v1.PLAUSIBLE_POSSESSION_S[1]
                    and v1.PLAUSIBLE_DEAD_SHARE[0] <= share["D"] <= v1.PLAUSIBLE_DEAD_SHARE[1])
        struct.append({"variant": k, "possessions": count, "mean duration (s)": "{:.1f}".format(mean_s),
                       **{c: "{:.1%}".format(share[c]) for c in "HAD"}, "plausible": "yes" if plaus[k] else "NO"})
    added = {b: int((t & (labels[b] != "D")).sum()) for b in ("E", "G")}

    os.makedirs(BOARD_DIR, exist_ok=True)
    for k in ("E", "G", "T_E", "T_G"):
        baseline.write_submission(os.path.join(BOARD_DIR, k + ".csv"), labels[k])
    out = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"), "--results", BOARD_DIR],
                         capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
    start = next(i for i, l in enumerate(out) if l.startswith("scoreboard"))
    end = next((i for i, l in enumerate(out) if l.startswith("macroF1 would order")), len(out))
    board = "\n".join(out[start:end]).rstrip()
    macro = {p[1]: float(p[2]) for p in (l.split() for l in out[start:end]) if len(p) > 3 and p[1] in labels}
    ok3 = {b: macro["T_" + b] >= macro[b] - 0.02 for b in ("E", "G")}

    gt = pd.read_csv(GT_PATH)["label"].values
    w = np.isin(gt, list("HAD"))
    pub = []
    for k in ("E", "G", "T_E", "T_G"):
        tr, pr = gt[w] == "D", labels[k][w] == "D"
        pub.append({"variant": k, "dead frames found": "{} of {} ({:.0%})".format(int((tr & pr).sum()), int(tr.sum()), (tr & pr).sum() / tr.sum()),
                    "called D that are D": "{:.0%}".format((tr & pr).sum() / max(pr.sum(), 1)),
                    "H/A/D frame accuracy": "{:.1%}".format((labels[k][w] == gt[w]).mean())})
    t_pub = t[np.arange(len(t))[w]]
    n_pub = sum(1 for s, e in st if s <= np.where(w)[0][-1] and e >= np.where(w)[0][0])

    prov = "\n".join([
        "- generated (UTC): {}".format(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python possession/outofplay_report.py`",
        "- git commit: `{}`{}".format(ms.git("rev-parse", "HEAD"), " (tracked files had uncommitted changes)"
                                      if ms.git("status", "--porcelain", "--untracked-files=no") else ""),
    ] + ["- possession/{} sha256: `{}`".format(f, ms.sha256(os.path.join(HERE, f)))
         for f in ("outofplay.py", "outofplay_report.py", "gapmodel.py", "gap_model.json")] + [
        "- input tracking sha256: `{}`".format(ms.sha256(asset_paths()["atd_raw"])),
        "- python {} / numpy {} / pandas {}".format(sys.version.split()[0], np.__version__, pd.__version__),
    ])
    yes = lambda ok: "**pass**" if ok else "**FAIL**"
    md = """# Out-of-play detector (T): dead ball while the ball is tracked

Generated by `possession/outofplay_report.py`. Rule pre-registered in `docs/worklog.md` ("2026-11-15 (out-of-play
detector, T)") before it was run. No learning. Aggregates only.

## Provenance

{prov}

## 1. Sample games (other matches, event labels)

{samp}

Total: {n_str} stretches; frame precision {prec:.2f}.

## 2. Our match

{n} stretches; length median {med:.1f} s, mean {mean:.1f} s, max {mx:.1f} s; {tot:.0f} s in total.
Frames newly written as `D`: {ae} on top of E, {ag} on top of G. {npub} stretch(es) touch the public window
({tpub} frames).

{struct}

## 3. Public 5 minutes (description and guard only)

{pub}

```
{board}
```

## 4. Decision rule (written in advance)

1. sample games: at least 20 stretches and frame precision >= 0.6: {n_str} stretches, precision {prec:.2f} -> {ok1}
2. plausibility: T_E {p_te}, T_G {p_tg}
3. public macroF1 not lower than the base by more than 0.02: T_E {mte:.4f} vs E {me:.4f} -> {o3e}; T_G {mtg:.4f} vs G {mg:.4f} -> {o3g}
4. the team agrees: **open**
""".format(prov=prov, samp=ms.pipe_table(pd.DataFrame(srows)), n_str=n_str, prec=prec, n=len(st),
           med=np.median(lens) if len(lens) else 0, mean=lens.mean() if len(lens) else 0, mx=lens.max() if len(lens) else 0,
           tot=lens.sum(), ae=added["E"], ag=added["G"], npub=n_pub, tpub=int(t_pub.sum()),
           struct=ms.pipe_table(pd.DataFrame(struct)), pub=ms.pipe_table(pd.DataFrame(pub)), board=board,
           ok1=yes(ok1) if measurable else "**not measurable** (fewer than 20 stretches)",
           p_te=yes(plaus["T_E"]), p_tg=yes(plaus["T_G"]), mte=macro["T_E"], me=macro["E"], o3e=yes(ok3["E"]),
           mtg=macro["T_G"], mg=macro["G"], o3g=yes(ok3["G"]))
    with open(RESULT_MD, "w") as handle:
        handle.write(md)
    print(md[md.index("## 1."):])


if __name__ == "__main__":
    main()
