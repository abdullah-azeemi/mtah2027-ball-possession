# possession/ - our own end-to-end possession baseline

## Current submission candidate: `submit.py`

`submit.py` is the new submission generator. It keeps T_E's dead-ball rules unchanged and changes
only the per-frame team evidence: a frame casts a vote for a team only when **exactly one team has a
tracked player within 1.5 m of the ball** (controlled possession). Contested, loose and untracked-ball
frames cast no vote and keep the previous possession, which is the challenge's own rule. The vote
window is +/-60 frames, and a tie keeps the previous decided possession rather than going to home. A ball moving
faster than 7 m/s is in flight (a pass, not control) and casts no vote either (`--flight-speed 0` turns this off).
`evidence.py` / `evidence_report.py` hold the pre-registered test of these rules on Metrica's sample games
(`docs/results/evidence_layer.md`).

```
python possession/submit.py --out <team_name>.csv          # the submission file
python possession/submit_report.py                          # validator + scorer + docs/results/submission.md
python possession/test_submit.py                            # 6 synthetic checks, no match data
```

On two other matches (Metrica sample games 1 and 2) the evidence rule beats T_E's nearest-player rule:
spellF1 0.575 / 0.635 against 0.521 / 0.488. Public 5 minutes, described only: **spellF1 0.5455, macroF1
0.7766, 16 predicted possessions against 17 true** (T_E: 0.3905 / 0.7567 / 25). The full-match score is
unknown. Method and rationale: the docstring of `submit.py`; evidence and caveats:
`docs/results/submission.md`.

The rest of this file documents the reference baseline that the pipeline was built on. T_E
(`make_submission.py`) remains in the repository as the previous candidate for comparison.

---

The first complete path from the tracking data to a scored submission file, written from scratch in
this repository, with the simplest method that is not a constant guess: **whoever is nearest the ball
has it, smoothed over time.**

It is the same method as the reference baseline in `starter/baseline.py` on purpose. Two reasons:

1. **A pipeline every later idea plugs into:** tracking, then per-frame labels, then the CSV, then the
   validator, then the official scorer.
2. **A check that we read the data the way the organisers do.** The labels produced here must equal,
   frame for frame, the committed `evaluation/results/baseline_*.csv`. They do: 0 differences in
   199,319 frames for all four variants, and the official scorer returns the published public-window
   scores. That confirms frame alignment, team assignment, units and the smoothing arithmetic.

## Method

| Step | What | Where |
|---|---|---|
| 1 | per frame with the ball tracked: the team of the tracked slot nearest to the ball (Euclidean, in metres, both axes scaled); no answer where the ball is not tracked | `baseline.nearest_team` |
| 2 | centred majority vote over +/- `window` frames (0, 25, 50, 125); `D` where the window holds no answer; ties go to the home side | `baseline.smooth_majority` |
| 3 | write `frame,label`, one row per video frame | `baseline.write_submission` |

No model, no training, no labels of any kind. The tracking (`tracking.py`) is loaded at 25 Hz for all
199,319 frames: positions in metres as float64 (to reproduce the reference's near-ties exactly), the
ball, and the team of each of the 46 track slots from the metadata XML (first team listed = home =
Barcelona).

## Run

```bash
python possession/baseline.py                       # writes the four variants, then checks them against the reference
python evaluation/validate_submission.py possession/output/submissions/baseline_smooth50.csv
python evaluation/run_evaluation.py --results possession/output/submissions   # official scorer, public 5 minutes
python possession/report.py                         # all of the above + a review, into docs/results/possession_baseline.md
```

Only `numpy` and `pandas` are needed (no PyTorch). Submissions are written to
`possession/output/submissions/`, which is git-ignored.

## What the review of the public window shows (descriptions, not rules)

`report.py` also describes where the baseline disagrees with the public ground truth, to understand
the labelling. The sample is tiny (17 possessions, 4 dead-ball stretches, 4 ball gaps of 1 s or more),
so these are leads to test on the full match with tracking-only evidence, not conclusions:

* **Dead ball is often tracked.** The ball is tracked on 66% of the ground-truth `D` frames, and on
  100% of two of the four dead-ball stretches (20.4 s and 14.5 s). The reference rule "ball not
  tracked means `D`" therefore finds only 314 of the 1,469 `D` frames and labels 902 of them `H` and
  253 `A`.
* **A missing ball is often live play.** Two of the ball gaps of 5 s or more (5.3 s and 9.3 s) lie
  entirely in live play (ground truth `A`); 164 `A` frames end up labelled `D`.
* **The nearest-player rule gets weaker with distance:** 90.6% right when the nearest tracked player is
  within 1 m of the ball, 82.6% at 1-2 m, 68.8% at 2-4 m, 62.0% at 4-8 m.
* **Too many possessions:** smooth50 gives 26 in the window against 17 true ones.

## Rules and limits

* The public ground truth is used for a correctness check and for description only. No threshold,
  rule or model choice is made from it (`RULES.md`, `EVALUATION.md`); the public window cannot rank
  anything.
* This baseline is the floor, not a candidate: it is the organisers' method, reproduced.
* Provenance for every report: command, git commit, SHA-256 of the code, tracking, metadata and
  public ground truth, library versions (`docs/results/possession_baseline.md`).

## Files

| File | Purpose |
|---|---|
| `tracking.py` | tracking for all frames as arrays, team of each slot |
| `baseline.py` | the two steps, the CSV writer, the comparison with the reference baselines |
| `report.py` | reproduction check, official scorer, public-window review, one results file with provenance |
| `deadball.py` | stoppage detector from the tracking alone: players' rolling speed below theta and the ball not moving, for at least T seconds |
| `v1.py` | the first full submission: nearest-team evidence, the dead-ball detector, a two-state Viterbi path with a 3 s switch penalty, label held through an untracked ball. Writes `possession/output/v1/possession_v1.csv` and `docs/results/possession_v1.md`. Result: under-segments on the public window (see the worklog); not shown to beat the reference |
| `make_submission.py` | **the one command that writes the submission** (T_E: variant E plus the out-of-play detector); see `docs/REPRODUCE.md` |
| `outofplay.py`, `outofplay_report.py` | out-of-play detector (dead ball while the ball is tracked beyond a line, until a restart kick); report in `docs/results/out_of_play.md`. Part of the submission |
| `gapscore.py`, `gapscore_report.py` | candidate F, a hand-weighted scorecard per ball-missing gap; `docs/results/gap_scorecard.md`. Not submitted |
| `gapmodel.py`, `gapmodel_report.py`, `gap_model.json` | candidate G, a logistic regression per gap trained on Metrica's open sample matches; `docs/results/gap_model.md`. Not submitted |
| `gap_check_list.py` | the 10 long gaps picked for the manual look (`docs/results/gap_check_list.md`) |
| `test_pipeline.py` | 10 synthetic checks of the building blocks, no match data needed |
| `v1b.py` | the reference baseline (window 50) with narrow changes (variants B to E; E plus the out-of-play detector is submitted), hold through an untracked ball and detector dead ball, with the four-variant ablation (A reference, B hold, C detector, D both). Writes `possession/output/v1b/possession_v1b.csv` and `docs/results/possession_v1b.md`. Not shown to beat the reference on the full match (which cannot be tested) |
| `deadball_report.py` | evaluates the detector on a pre-registered grid against Smart Tagging restarts, writes `docs/results/deadball_detector.md` |
| `output/` | submissions (local, git-ignored) |
