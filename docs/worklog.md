# Work log

A dated record of what was done, why, and what was found, so a reviewer can follow the reasoning
and re-check any number. Newest entry first. Each finding says how it was obtained; anything not
yet reproducible from code in this repository is marked **ad hoc**.

## 2026-11-16 (random forest + SHAP result) - RF does not replace CS; SHAP shows the rule is the model's main signal

`python possession/rf_report.py` -> `docs/results/rf_shap.md` (needs scikit-learn and shap; not in the submission path).

| # | Finding | Evidence |
|---|---|---|
| 73 | Held-out spellF1, RF vs CS: 0.5717 vs 0.5748 (game 1), 0.6405 vs 0.6345 (game 2). Not higher on both, so **not adopted**; differences are inside the bootstrap bands | `docs/results/rf_shap.md` |
| 74 | **RF without the rule's label as a feature is clearly worse than the rule:** spellF1 0.5415 / 0.5545 (CS 0.5748 / 0.6345), raw frame accuracy 0.83 / 0.82 (with the rule feature 0.91 / 0.93), and it fragments (337 / 323 possessions); with the ball blanked it falls to 0.41 / 0.36. On two training matches, a forest does not rediscover the rule from raw distances | same |
| 75 | SHAP (2,000 held-out live frames per game): with the rule feature, the rule's label dominates (mean abs SHAP 0.23 / 0.20, next feature 0.06 / 0.08). Without it, the top signals are each team's nearest distance to the ball (raw and +/-12-frame mean) and the number of each team's players within 1.5 m: the same quantities the rule uses. Ball speed ranks about tenth by mean abs SHAP; it matters on fewer frames (passes), where it decides the vote | same |

**Decision.** Keep CS. For the write-up: two learned models (gradient boosting, random forest) trained on other matches
and tested leave-one-game-out tie with the rule when given it and lose without it; SHAP shows they rely on the rule's
own signals. This is the evidence for using an interpretable rule rather than a trained model.

## 2026-11-16 (random forest + SHAP, RF) - Pre-registration, written before any model was fitted

Same data, reference, features, smoothing and leave-one-game-out protocol as GB (below).
`sklearn.ensemble.RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=0)`, no
tuning. **Decision rule:** identical to GB's (spellF1 above CS on both held-out games, macroF1 not lower by > 0.01).
**SHAP (description only, chooses nothing):** `shap.TreeExplainer` on each held-out game, 2,000 live frames drawn with
seed 0; report mean |SHAP| per feature for p(home). Also fit a second RF **without** the `cs_label` feature and report
its SHAP ranking, to see which raw tracking signals a learned model relies on and whether they match the rule's
(distance to the ball, one-team control, ball speed).

## 2026-11-16 (gradient-boosted evidence result) - Run as pre-registered; GB does not replace CS

`python possession/gbm_report.py` -> `docs/results/gbm_evidence.md` (needs scikit-learn; not in the submission path).

| # | Finding | Evidence |
|---|---|---|
| 70 | Held-out spellF1, GB vs CS: test game 1 **0.5832** vs 0.5748 (+0.008), test game 2 **0.6320** vs 0.6345 (-0.003). Not higher on both games, so GB is **not adopted**. Both differences are far inside the bootstrap bands (about +/-0.05) | `docs/results/gbm_evidence.md` |
| 71 | GB is slightly better on the frame view and on transitions: macroF1 +0.005 / +0.003, transF1@25 +0.050 / +0.014, team accuracy +0.008 / +0.004; with the ball blanked ~40% it is within 0.01 of CS on both games | same |
| 72 | Why it does not move spellF1: both methods predict far fewer possessions than the reference (GB 237 / 218, CS 246 / 234, true 381 / 363). The limit is segmentation (missed short possessions), not the per-frame evidence: a classifier that is right on ~91-93% of held-out live frames, smoothed by the same +/-60 window, gives the same spells | same |

**Decision.** Keep CS (rules only, numpy/pandas). Report GB in the technical document as a negative result: a learned
evidence layer, trained on other matches with a leave-one-game-out test, matched the rule but did not beat it. The
next gain, if any, is in segmentation (short possessions), not in the frame evidence.

## 2026-11-16 (gradient-boosted evidence, GB) - Pre-registration, written before any model was fitted

**Question.** Does a learned per-frame classifier beat the rule now submitted (CS: one team within 1.5 m and the ball
slower than 7 m/s, +/-60 vote) at deciding which team has the ball while it is in play?

**Data.** Metrica open sample games 1 and 2 only, with the event-derived reference of the evidence-layer test
(`possession/evidence_report.py`). **Leave one game out:** train on game 1, test on game 2, and the reverse. Nothing
from our match is used to fit or choose. Training rows: live reference frames (H or A) inside the halves.

**Features per frame (tracking only, no pitch position, so the change of ends does not matter):** for each team the
distance of its nearest tracked player to the ball (capped at 30 m) and the number of its players within 1.5, 3 and
5 m; ball speed (capped at 40 m/s) and a ball-tracked flag; seconds since the ball was last tracked (capped at 30 s);
the same distances and counts averaged over +/-12 frames; and the CS rule's own team label for the frame (so the model
can learn when to overrule it). Missing values stay NaN (the model handles them).

**Model.** `sklearn.ensemble.HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)`,
defaults otherwise, no tuning. Output p(home) per frame; segmentation: centred mean of p over +/-60 frames, home if
above 0.5; then the reference `D` (oracle dead ball, as in the evidence-layer test).

**Decision rule.** GB replaces CS only if, on the held-out game, spellF1 is higher than CS's on **both** games and
macroF1 is not lower by more than 0.01 on either. The dropout condition (ball blanked ~40%, seed 0) is reported; a
loss of more than 0.02 spellF1 there on both games is a flag. If GB wins, it would also need a full-match plausibility
check on our match and a new dependency (scikit-learn) in the submission environment before adoption.

## 2026-11-15 (evidence layer result) - Run as pre-registered; controlled-ball and in-flight gate kept, fallback and reception switch dropped

`python possession/evidence_report.py` -> `docs/results/evidence_layer.md`. Condition 1 (ball as tracked, oracle `D`):

| # | Finding | Evidence |
|---|---|---|
| 63 | **N -> C (ahmed's gate) kept:** spellF1 0.521 -> 0.562 (game 1), 0.488 -> 0.599 (game 2); macroF1 0.879 -> 0.890, 0.863 -> 0.899. First evidence for the gate that does not come from the public window | `docs/results/evidence_layer.md` |
| 64 | **C -> CS (ball in flight, 7 m/s) kept:** spellF1 0.562 -> 0.575, 0.599 -> 0.635; macroF1 0.890 -> 0.933, 0.899 -> 0.950; team accuracy 0.836 -> 0.899, 0.848 -> 0.925 | same |
| 65 | **Fallback (CSF) and switch on reception (CSR, CSFR) dropped:** each rises on game 1 and falls on game 2 (CSR 0.611 / 0.602 against CS 0.575 / 0.635). Not consistent, so not kept | same |
| 66 | Robustness (ball blanked ~40% in long stretches): ordering N < C <= CS holds on both games (CS 0.462 / 0.473, N 0.394 / 0.379); no step flagged | same |
| 67 | Our match with T_E's dead ball: CS gives 334 possessions of mean 10.9 s (C 395 / 9.2 s, T_E 493 / 7.4 s), `D` 42.7% unchanged; the gate removes 5,175 of 36,876 controlled votes. `H 53973 A 37061 D 67823 X 40462`, SHA-256 `8409a8920f34f0456d68aefdf47b9640804625173bc02d0ec866a2ca870870e5`, validator passes | `python possession/submit.py --out ...` |
| 68 | Public window, description only, after the choice: CS macroF1 0.7766, spellF1 0.5455, acc 0.8127 (C 0.7715 / 0.5212 / 0.8065; T_E 0.7567 / 0.3905 / 0.7875) | `evaluation/run_evaluation.py` |
| 69 | Review fixes in `submit.py`: the vote now runs inside each half (window and tie memory no longer cross half-time; output bytes unchanged, `--flight-speed 0` reproduces ahmed's hash `161bc1a3...`); `controlled_codes` docstring no longer claims an off-pitch check it does not do | `python possession/test_submit.py` (6 checks) |

**Decision.** Candidate = controlled-ball evidence + in-flight gate (CS) + T_E's dead-ball rules, written by
`possession/submit.py`. Caveats: the reference possession is built from event data (start-of-event team, held), not
frame-annotated; two sample matches only; their tracked ball is cleaner than ours. The full-match score on our match
is still unknown.

## 2026-11-15 (evidence layer on sample matches) - Pre-registration, written before any variant was run

**Why.** The choice between T_E's evidence (nearest player at any distance) and the controlled-ball evidence of
`possession/submit.py` was made on the public 5 minutes (finding 62), which cannot rank two files. This test uses an
independent set: Metrica Sports' open sample games 1 and 2 (other matches, allowed by the FAQ), with a possession
reference built from their event data. Nothing from our match or its public window is used to choose.

**Reference labels (sample games 1 and 2).** Possession-defining events are PASS, SHOT, RECOVERY, SET PIECE, BALL LOST
and BALL OUT; each gives its `Team` the ball from its start frame until the next such event (a loose ball after BALL LOST
stays with that team until the other team's RECOVERY, the challenge's rule). CHALLENGE, CARD and FAULT RECEIVED are
ignored. Dead frames come from `gapmodel._dead_from_events` (previous event end to the restart); outside the halves `X`.
**The same reference `D` is given to every variant** (oracle dead ball), so only the H/A evidence layer is compared.

**Variants (parameters fixed here, no grid):**

| id | evidence per frame | segmentation |
|---|---|---|
| N | nearest tracked player's team, any distance (T_E, reference) | centred vote +/-50, ties home, hold |
| C | exactly one team within 1.5 m (ahmed) | centred vote +/-60, ties previous, hold |
| CS | C, and the ball is slow: speed over 0.16 s (centred, 4 frames) under **7.0 m/s**; faster balls are in flight and vote for nobody | as C |
| CSF | CS; where no CS vote lies within +/-75 frames (3 s), a slow ball votes for the nearest team (fallback) | as C |
| CSR | CS | **switch on reception**: runs of same-team votes (ignoring no-vote frames); runs with fewer than 5 votes are dropped (deflections); a possession starts at the first vote of a kept run; hold between |
| CSFR | CSF | as CSR |

7.0 m/s: a dribbled ball moves with the player (rarely above 7 m/s); a pass or clearance is faster. 5 votes = 0.2 s
of control. 3 s fallback: longer than a typical pass flight.

**Conditions.** (1) primary: the tracked ball as given. (2) secondary, robustness: the ball blanked in alternating
stretches (tracked ~ exponential mean 15 s, missing ~ exponential mean 10 s, seed 0), close to our match's missingness.

**Metrics.** Official `evaluation/possession_metrics.evaluate`: spellF1 (primary), macroF1, transition F1 at 25 frames,
predicted vs true possession count; block-bootstrap interval for spellF1 (200 draws).

**Decision rule.** A step (N->C, C->CS, CS->CSF, CS->CSR, CSF->CSFR) is kept only if, in condition 1, spellF1 rises on
**both** games and macroF1 falls by no more than 0.01 on either. A kept step that loses more than 0.02 spellF1 on both
games in condition 2 is flagged and not adopted without a team decision. The final evidence layer is put on top of T_E's
dead-ball rules for our match, checked for plausibility (possessions, `D` share unchanged); its public-window score is
reported only as a description, after the choice.

## 2026-11-15 (controlled-ball candidate) - New candidate on `ahmed/ball-tracking`: gated per-frame evidence

Method: `possession/submit.py`. Only the per-frame team evidence changes from T_E: a frame casts a vote only when
**exactly one team has a tracked player within 1.5 m of the ball** (controlled possession); contested, loose and
untracked-ball frames cast no vote and keep the previous possession. The vote is a centred majority over +/-60 frames,
and a tie keeps the previous decided possession rather than going to home. Dead-ball rules (stoppage detector,
second-half pre-kick-off, 10 s ball gaps, out-of-play) are exactly T_E's and unchanged. No learning, no randomness.

| # | Finding | Evidence |
|---|---|---|
| 58 | The reference's ungated nearest-player vote gives 25 predicted possessions on the public window against 17 true. Fixing dead ball alone barely moves spellF1 (T_E with oracle `D`: 0.4000), so the loss is team segmentation | ad hoc scorer runs on the public window |
| 59 | Gated evidence at 1.5 m with a +/-60 window: public spellF1 **0.5212**, macroF1 **0.7715**, 16 predicted possessions against 17 true; team accuracy 0.8858 (T_E 0.8620), fragmentation 0.86 (T_E 1.29) | `python possession/submit_report.py`, `docs/results/submission.md` |
| 60 | Diagnostics: ground-truth live frames + ahmed's dead frames score 1.0000, so every `D` ahmed writes on the window is correct; ahmed's live labels + ground-truth dead frames raise spellF1 to 0.5818, so dead-ball recall (0.481, unchanged from T_E) is still a loss | ad hoc scorer runs |
| 61 | Full match, tracking only: 395 possessions (H 196 of mean 11.0 s, A 199 of mean 7.4 s) and `D` 42.7%, against T_E's 493; `H 54077 A 36957 D 67823 X 40462`, SHA-256 `161bc1a36679f0e8d5e29581fe20862d5024206763371c15b03bb2d48d955d61`, validator passes | `python possession/submit.py --out ...`; `evaluation/validate_submission.py` |
| 62 | The public window has 17 possessions and its scores swing by more than 0.2 on small parameter changes. The chosen settings sit on a plateau (gate 0.8 to 3.0 m and window 50 to 70 all give about 0.49 to 0.52), not at a grid maximum, but no full-match ground truth means the gain is unverified | ad hoc grid; section 7 of `docs/technical_document.md` |

**Decision.** `submit.py` is the submission candidate; T_E stays in the repository for comparison. Tests:
`python possession/test_submit.py` (5 synthetic checks) and `python possession/test_pipeline.py` (10) pass.
**Disclosure:** the public-window score informed the choice between T_E and this candidate. That is a deviation from
using the window only as a check; the technical document states it. Thresholds were chosen on a plateau, not tuned to
the window's maximum.

## 2026-11-15 (adoption of T_E) - The team adopts T_E as the submission, as a documented deviation

**Decision (team, 2026-11-15).** The submission candidate becomes T_E ("v1f"): variant E plus the out-of-play detector T.
`possession/make_submission.py` now writes T_E only: 199,319 rows, H 52,574, A 38,460, D 67,823, X 40,462; SHA-256
`fe38ac194ba507cdf8cc3ed4e2da26e907a2e4b65bf006dc7320f66131e5acb9` (two runs identical, equal to the T_E file of
`outofplay_report.py`, validator passes, 493 possessions of 7.4 s).

**This deviates from the pre-registration.** T failed criterion 1 (sample-game precision 0.59 against a bar of 0.6).
The team's reasons, written after seeing the results: games 2 and 3 blank the ball during dead play and so test only
false alarms; game 1, the one sample match with tracked dead ball like ours, gives 0.73; T_E stays inside the
plausibility bounds (D 42.7%). The public window was not a reason (one stretch). The deviation is stated in the
technical document. F and G are kept as documented, not-submitted candidates; their code and results stay in the
repository (`gapscore*.py`, `gapmodel*.py`, `gap_model.json`, `docs/results/gap_scorecard.md`, `gap_model.md`).

## 2026-11-15 (out-of-play result) - Detector T run as pre-registered; criterion 1 fails narrowly, public window improves

Run: `python possession/outofplay_report.py` (tables in `docs/results/out_of_play.md`). Code `possession/outofplay.py`;
one synthetic check added (10 pass). `make_submission.py` unchanged (E by default, G with `--variant G`).

| # | Finding | Evidence |
|---|---|---|
| 54 | Sample games: 29 stretches, frame precision 0.59 (game 1: 18 stretches, precision 0.73, finds 39.9% of its 8,717 tracked dead frames; game 2: 4, 0.31; game 3: 7, 0.00) | `out_of_play.md` section 1 |
| 55 | The false stretches in games 2 and 3 are of two kinds (ad hoc look, same data): the ball tracked beyond a line for 2.6 to 9.4 s while the events show live play (5 in game 3, 2 in game 2), and two stretches of 48 to 49 s where the ball went out and then stayed untracked, so no restart was seen and the stretch ran towards the 60 s cap through live play | ad hoc |
| 56 | Our match: 39 stretches, median 12.2 s, 602 s in total. On top of E it writes 7,560 frames newly as `D` (T_E: D 42.7%, 493 possessions of 7.4 s, plausible); on top of G 6,526 (T_G: D 45.8%, **outside** the 45% bound) | section 2 |
| 57 | Public window (one stretch, 342 frames, touches it): dead frames found E 25% to T_E 48%, all still correct; H/A/D frame accuracy 74.2% to 78.7%; macroF1 0.6613 to 0.7567, spellF1 0.3707 to 0.3905. One stretch is one event: this is not evidence of a general gain | section 3 |

**Decision by the written rule: T is not adopted.** Criterion 1 fails (precision 0.59 < 0.6) and T_G also fails
plausibility. Criterion 3 passes for both. Two things for the team, stated as facts and not as a re-reading of the rule:
games 2 and 3 have almost no tracked dead ball (finding 54 / pre-registration), so they test only false alarms; and the
one sample game that resembles ours (game 1) gives 0.73. Adopting T_E anyway would be a **documented deviation** from
the pre-registration, and the technical document would have to say so. Any fix of the 60 s cap (finding 55) is a new
rule and needs a new entry.

## 2026-11-15 (out-of-play detector, T) - Pre-registration, written before the detector was run

**Why.** In the public window about two thirds of the dead frames have the ball tracked (a player carries it back to a
throw-in, corner or goal kick), and no gap rule can see them: E finds 25% of the public dead frames, G 34% (ad hoc
count, 2026-11-15). Earlier: the ball is tracked outside the pitch on 2.1% of graded frames in 71 runs, and such a run
comes before 13 of 26 throw-in tags and 5 of 10 goal-kick tags (finding 43).

**Why no learning.** In the sample games dead ball is almost never tracked (share of dead frames with the ball tracked:
16.8%, 3.5%, 0.2% in games 1 to 3, against about two thirds in our public window), so they cannot teach this. The rule
below is fixed from the Laws of the Game (ball wholly over a line is out) and physical sense, not from any label.

**Rule.** Ball speed `v` = |ball(t) - ball(t-1)| x 25, then a centred median over 10 frames (0.4 s).
1. *Out:* the ball is tracked more than 0.5 m beyond a touchline or goal line on at least 10 consecutive frames (0.4 s).
2. *Dead from* the first frame of that run.
3. *Until the restart:* the first frame t, at least 2 s after the out run ends, where the ball is tracked within the
   pitch (0.5 m margin) and `v` >= 6 m/s (a throw or kick; carrying is slower). If no restart is found within 60 s of
   the out run's start, the stretch ends at 60 s.
Untracked frames inside a stretch stay dead. Only graded frames are written. Variant T_E = E plus these stretches as
`D`; T_G = G plus these stretches.

**What will be reported** (`docs/results/out_of_play.md`, aggregates only): on each sample game, stretches found,
precision of their frames against the event labels, share of tracked dead frames found; on our match, stretches, their
length, added `D`, structure of T_E and T_G; the public window for E, G, T_E, T_G as a description.

**Decision rule.** T is added to the candidate (on top of whichever of E or G the team picks) if (1) on the sample games
the detector fires at least 20 stretches in total and their frame precision is at least 0.6; if it fires fewer than 20,
criterion 1 is "not measurable" and is reported so, not as a pass; (2) plausibility holds (D 5% to 45%, mean
possession 5 to 40 s); (3) public macroF1 not lower than the base variant's by more than 0.02; (4) the team agrees.
No parameter above is changed after the run without a new dated entry.

**Caveats.** A ball boy throwing a new ball in moves at more than 6 m/s while play is dead, and ends a stretch early.
Tracking noise near a line can make a live ball look out; the 0.5 m margin and 0.4 s minimum are the guard. Free
kicks and fouls (ball stays inside) are not covered; they remain the speed detector's job.

## 2026-11-15 (learned gap model result) - Variant G run as pre-registered; criteria 1 to 3 pass, one strong caveat

Run: `python possession/gapmodel_report.py` (tables in `docs/results/gap_model.md`; weights in `possession/gap_model.json`).
`make_submission.py` still writes E.

| # | Finding | Evidence |
|---|---|---|
| 48 | Sample games: 476 gaps, 238 dead. Dead share is 26 to 41% for gaps under 10 s and 73 to 84% for gaps of 10 s or more | `gap_model.md` section 1 |
| 49 | Held out (leave one game out), mean frame F1 of dead: G 0.874, scorecard without tag 0.859, E rule 0.826; gap F1 0.849, 0.821, 0.684 | section 2 |
| 50 | The fitted model is essentially length and players' speed: weights per sd log_len +1.95, speed -0.72, edge_out -0.12, restart_in -0.01, keeper +0.05. The keeper pattern fires on 1.5% of sample gaps and gets no weight, so G sends no gap to a keeper | section 3 |
| 51 | Shift: our gaps are much longer (mean ln length 2.08 vs 0.83; the sample games have 71 to 84 gaps under 5 s per game, ours 18). Our tracker seems to bridge short losses itself | section 4 |
| 52 | On our match G writes D on 121 of 150 gaps (43 short, 78 long); D share 41.7% (E 37.9%); 480 possessions of mean 7.7 s; E to G relabels 6,558 graded frames, almost all H/A to D. Public window: macroF1 0.6697 (E 0.6613), spellF1 0.3619 (E 0.3707), both inside the noise | sections 4 and 6 |
| 53 | **Post hoc diagnostic (not part of the decision rule).** Restricted to sample gaps of 5 s or more (the range our match has), held-out frame F1: G 0.898 / 0.868 / 0.868, "every gap is D" 0.896 / 0.871 / 0.882, E rule 0.863 / 0.836 / 0.804 (games 1 / 2 / 3). In that range the learned model does no better than calling every gap dead; its gain over E comes from writing D on more gaps | ad hoc, same code |

**Status.** Criteria 1 to 3 pass; criterion 4 (team) is open. Finding 53 means the honest summary is: on other matches,
most ball gaps of 5 s or more are dead ball, so E's 10 s threshold is too high. G is a learned way of saying that,
not evidence for the keeper or restart features.

## 2026-11-15 (learned gap model, v1e) - Pre-registration, written before any model was fitted

**Why.** The scorecard's weights are set by hand. External data about other games is allowed (FAQ), so the weights
can be learned from labelled matches instead. Data: Metrica Sports' open sample data (`github.com/metrica-sports/
sample-data`, three anonymised matches with tracking and event data, 25 Hz; README: free to use, acknowledge the source),
downloaded to `data/external/metrica_sample/` (git-ignored) with the user's approval. Seen before writing this: the ball
is missing on 39.1% / 41.0% of frames of games 1 / 2 in 155 / 169 gaps (our match: 40.7%, 150 gaps), and the event types.

**Labels (sample games only).** A restart is a `SET PIECE` event, or any event whose subtype starts with `GOAL KICK`
(games 1 and 2 log goal kicks as passes). Dead ball runs from the end frame of the last event before the restart
(ignoring `CARD` and `SET PIECE` events) to the restart's start frame. Only frames between a period's first and last
event count. A gap is **dead** when more than half of its frames are dead.

**Features** (same code for the sample games and our match; nothing from Smart Tagging or events, because Smart Tagging
does not exist for the sample games and events are the labels): `log_len` = ln(gap length in s); `speed` = median over
the gap of the 3 s rolling median player speed, speed computed from position differences in both data sets (clipped at
12 m/s); `edge_out`, `restart_in`, `keeper` as in the scorecard (`keeper` without the tag condition).

**Model.** Logistic regression on the standardised features, L2 penalty 1.0, fitted by Newton's method in numpy. A gap
with predicted probability of dead >= 0.5 is `D`. A gap predicted live with `keeper` true goes to the keeper's team;
other live gaps keep variant D's labels (hold). Everything else in the pipeline is unchanged. This is variant G (v1e).

**Evaluation.** Leave one game out over the three sample games: fit on two, test on the third. Report for each test game
the F1 of "dead" over gaps and over gap frames (frame-weighted), for G and for two fixed rules on the same gaps:
E's rule (gap of 10 s or more is dead) and the scorecard without its tag feature. Then fit on all three games, print the
coefficients, apply to our match and report the same aggregates as the scorecard plus the public window.

**Decision rule.** G replaces E as the submission candidate if (1) the mean frame-weighted dead F1 of G over the three
held-out games is at least E's rule's + 0.02; (2) plausibility bounds hold on our match (D 5% to 45%, mean possession
5 to 40 s); (3) public-window macroF1 of G is not lower than E's by more than 0.02; (4) the team agrees. Otherwise E
stays. Nothing above is changed after seeing the held-out results without a new dated entry.

**Caveats.** Three games is little. The sample data's tracking may lose the ball for other reasons than our match's
(a different camera or tracker); that shift cannot be measured without our labels. Event timing defines the dead
stretches only approximately.

## 2026-11-15 (gap scorecard result) - Variant F run as pre-registered; criteria 1 to 3 pass, team review open

Run: `python possession/gapscore_report.py` (tables in `docs/results/gap_scorecard.md`). Code `possession/gapscore.py`;
two synthetic checks added to `possession/test_pipeline.py` (8 pass). The rule was run unchanged.
`make_submission.py` still writes E (SHA-256 `cda670e1...943b9`, unchanged).

| # | Finding | Evidence |
|---|---|---|
| 44 | 150 gaps. Decisions: 89 `D` (71 long, 18 short), 48 short gaps kept (hold), 13 `keeper` (8 long, 5 short). Features fire on: long 79, slow 46, edge_out 43, restart_in 44, tag 45, keeper 13 | `gap_scorecard.md` section 1 |
| 45 | E to F relabels 6,659 graded frames (4.2%): D to A 4,367, H to D 1,279, A to D 552, D to H 268, H to A 193. F: 477 possessions of mean 8.5 s; H 34.8%, A 29.0%, D 36.2% (E: 493, 8.0 s, D 37.9%) | section 2 |
| 46 | Decision rule: (1) keeper on 13 gaps, inside 1 to 15: pass; (2) plausibility: pass; (3) public macroF1 F 0.6610 vs E 0.6613: pass (spellF1 equal, 0.3707). (4) team review: **open** | section 3 |
| 47 | 11 of the 13 keeper gaps go to Athletic (A), 2 to Barcelona. Five keeper gaps last 16 to 70 s, and one of them (57:44, 69.5 s) is also `slow`. A keeper may hold the ball in his hands for only 6 s (Laws of the Game, law 12), so a keeper "holding" the ball for 20 s or more is unlikely to be live play; more likely a restart the keeper takes from inside his box (a free kick or goal kick placed outside the goal area) that the tags miss. The pre-registered rule has no length limit on `keeper`; that is a design gap noticed after the run | section 5 |

**Status.** By the written rule F cannot replace E until the team reviews the per-gap table (criterion 4). Finding 47
is a reason to look hard at the long keeper gaps. Any limit on keeper gap length would be a **post hoc change**,
and would need a new dated pre-registration that says it was prompted by this result. Note: the 57:44 gap is one of
the 10 gaps described by a team member (a free kick). That description was not used to judge F; finding 47 rests on
law 12 and on the `slow` feature.

## 2026-11-15 (gap scorecard, v1d) - Pre-registration, written before any gap feature was computed

**Why.** Variant E writes `D` on every ball-missing run of 10 s or more and holds the last team on shorter runs. Two
weak points (handover note, 2026-11-14): (1) a keeper holding the ball in his hands is *live* play (the keeper's team
has possession) even though the ball is often untracked for 10 s or more, so E would wrongly write `D` there; (2) the
53 gaps of 5 to 10 s are all held, and none was ever checked. Before this was written, the only things computed were
the gap-length histogram (150 gaps in the graded halves: 18 under 5 s, 53 of 5 to 10 s, 79 of 10 s or more) and the
count of gaps ending near a Smart Tagging set-piece start (5 of 53 short gaps, 12 of 79 long ones). A team member also
described the same 10 long gaps of the earlier gap check again in chat (same events as the 2026-11-12 entry). Those
descriptions are **not** used as labels, to set a threshold, or to validate (FAQ: no hand labels on the game).

**Unit.** A gap is a maximal run of frames with the ball untracked, inside one graded half. `s` is its first frame,
`e` its last, `L` its length in seconds; `s-1` is the last frame with the ball seen, `e+1` the first frame it is seen
again (a side is "unknown" if that frame falls outside the half).

**Features** (pitch 105.3 x 68.0 m, origin at a corner; penalty area 16.5 m deep, 40.32 m wide; goal area 5.5 m deep,
18.32 m wide; both centred on y = 34):

| feature | definition |
|---|---|
| `long` | L >= 10 s |
| `slow` | median over the gap of the 3 s rolling median player speed (`deadball.rolling_player_speed`) < 1.0 m/s (the detector's theta) |
| `edge_out` | ball at `s-1` within 2 m of a touchline or goal line, or outside the pitch |
| `restart_in` | ball at `e+1` at a restart spot: within 3 m of a corner flag, within 3 m of the centre spot, inside a goal area widened by 1 m, or within 1 m of (or beyond) a touchline |
| `tag` | a Smart Tagging `SET PIECES` tag starts in [s - 3 s, e + 3 s] |
| `keeper` | all of: no `tag`; ball at `s-1` and at `e+1` both inside the same penalty area; that penalty area belongs to team T (the team whose tracked slots have the mean x closer to that goal, over the half); at `e+1` the ball lies outside the goal area and within 2 m of T's deepest tracked slot (the slot of T nearest its own goal line) |

**Score.** points = 2·long + 2·slow + 1·edge_out + 1·restart_in + 2·tag − 3·keeper.
**Decision.** A gap with points >= 2 is `D`. A `keeper` gap is labelled with team T (H or A) on all its frames,
overriding the stoppage detector inside the gap. Every other gap keeps what variant D gives it (hold, or the detector's
`D`). Everything else in the pipeline (vote, hold, detector, second-half pre-kick-off `D`, `X` outside the halves) is
unchanged. This is variant F (v1d). Without `keeper`, a gap of 10 s or more always scores at least 2, so F differs
from E only through (a) `keeper` gaps and (b) short gaps that score 2 or more.

**What will be reported** (`docs/results/gap_scorecard.md`, aggregates only): number of gaps per score and per
feature; per-gap table (frames, video time, length, features, points, decision, no coordinates); frames relabelled E to
F; possessions, mean length and D share of E and F; the public-window scores of E and F as a description.

**Decision rule (written now).** F replaces E as the submission candidate only if all of these hold:
(1) `keeper` fires on 1 to 15 gaps (0 means the feature does nothing; more than 15 means it is too loose for a match
with about 80 long gaps); (2) the D share of F stays within 5% to 45% and the mean possession within 5 to 40 s (the
earlier plausibility bounds); (3) public-window macroF1 of F is not lower than E's by more than 0.02 (a guard against
breaking something, not a target: nothing is changed after seeing it); (4) the team reads the per-gap table and agrees.
If (1), (2) or (3) fails, E stays and the result is reported as negative. No point value, distance or threshold above
will be changed after the run without a new dated pre-registration.

**Caveats written in advance.** The deepest slot is not always the keeper (slots are tracklets; a defender can be
deeper for a moment). The keeper rule cannot see a keeper who punts the ball away before it is tracked again (the ball
then reappears far from him), so it is conservative. Short-gap evidence is weak: the tags cover few restarts, and
throw-ins and goal kicks often restart without the players slowing.

## 2026-11-12 (v1c result) - Variant E run as pre-registered; it becomes the main candidate

Run: `python possession/v1b.py` (table in `docs/results/possession_v1b.md`). Variant A still reproduces the reference
(0 frames differ). Variant E: 493 possessions of mean 8.0 s; H 35.6%, A 26.5%, D 37.9%. D to E relabels 27,292 graded
frames; A to E 15,895 (10.0%). Plausibility bounds (5 to 40 s, D 5% to 45%) hold; the failure criterion (D above 45%)
did not trigger; the validator passes. By the decision rule E replaces D as the candidate. Public window (description
only): E spellF1 0.3707, macroF1 0.6613; A 0.3256, 0.6207; B and D 0.3610, 0.5157. These were not used for the choice
(E was chosen before the run) and with 17 possessions they sit inside the noise.
New file: 60,263 D, SHA-256 `cda670e1a009a70009954a2701b7fd2cea08985b7c4b5403a4374174c22943b9`.
Caveat: 37.9% D is high; if long gaps contain live play, E loses on those frames. Unknown without full-match truth.

## 2026-11-12 (gap check, v1c) - Manual look at 10 long ball gaps, and a pre-registered change (written before it was run)

**What was done.** `possession/gap_check_list.py` picked 10 ball-missing gaps of at least 10 s, evenly spaced by start
frame (rule fixed before running; list in `docs/results/gap_check_list.md`). A team member watched the tactical video
for each and described what was happening. This is a **human inspection of the match video used for a design decision**.
It was not used to write labels. It is disclosed in the technical document (section 10) so that reviewers can judge
whether it is compatible with the rule against hand annotation.

**What was seen (member's description).** In all 10 gaps play was not live: keeper holding the ball / goal kick (1, 6, 7),
corner set-up (2), drinks break (3), penalty stoppage (4), foul and free kick (5), goal celebration (8), yellow card (9),
a stoppage (10, reported as a scuffle). In none was the ball in play but hidden. Consequences:

* A ball detector from video has nothing to find in these gaps. Dropped; not built.
* Hold (variant B/D) assumes possession carries across a missing-ball stretch. For long gaps this is probably wrong:
  the stretch is mostly dead ball. The reference's "ball missing = D" is probably right for long gaps.
* Caveats: n = 10, so this is evidence, not proof. Public-window gaps were shorter, and there 164 live frames were
  wrongly D (finding in the baseline entry), so short gaps may be live play.

**Pre-registered variant E (v1c).** Variant D plus: where the ball is untracked for a run of at least 250 frames (10 s)
inside a graded half, write D over that run. Shorter untracked runs keep the hold. Nothing else changes. The 10 s
threshold was fixed here, before running, as the shortest length in the inspected sample (11.9 s rounded down to a
round number); it is not tuned.

**Reported for E, with A to D, with equal prominence:** possessions, mean duration, share of H/A/D, frames relabelled
relative to D, the validator, and the public-window scorer (a description only; nothing is chosen from it).

**Decision rule (set now).** E becomes the main candidate if the plausibility bounds written earlier hold (mean
possession 5 to 40 s, D share 5% to 45%) and the validator passes. The choice rests on the reasoning above, not on a
score; the full-match score of E is unknown, as for the others.

**Failure criterion.** If E's D share exceeds 45% the 10 s rule is too aggressive and D stays the candidate.

## 2026-11-12 (decision) - Main candidate: v1b (hold + detector); submission pipeline made reproducible

**Decision (the team's, taken after finding 50):** the main submission candidate is **v1b, variant D** (hold plus
detector dead ball on the reference's window-50 smoothing). Variant B (hold only) and the reference are the
alternatives. The choice is a judgment about risk, not a result: no available evidence ranks them (finding 50).
The lean towards D rested on its D share (20.8%) being close to the only available estimate of a live match's dead
share (the public window's 19.6%), which is weak support.

**Done**

* `possession/make_submission.py`: one command that writes the file and re-checks the format. Output for this
  data: 199,319 rows (H 70,891, A 54,995, D 32,971, X 40,462), SHA-256
  `9697b068cf45641d36db6cf60dfe10b9c47c4c4d8b6e7e82cb04c21fae71f587`; the validator accepts it (526 possessions of
  mean 9.6 s counting each `X` as a gap).
* Reproducibility checked: two runs and the earlier generator give the same bytes; the same bytes with numpy
  2.4.0, 2.4.1 and 2.4.2 in clean virtual environments with only numpy and pandas. numpy 2.4.0 has since been
  yanked on PyPI, so `requirements-submission.txt` pins 2.4.1.
* `possession/test_pipeline.py` (6 checks on synthetic data), `docs/REPRODUCE.md`, `docs/technical_document.md`.

**Open: data handling and the rule on hosted LLMs (to resolve with the organisers before submitting).**
`OVERVIEW.md` allows LLMs anywhere in the pipeline but states that participants shall not upload, disclose or make
available to any third-party hosted or cloud LLM any videos, datasets or other restricted materials provided for the
competition, or any extracts of them. This work was done with Claude Code, a hosted LLM agent. The scripts ran on
the local machine, but their printed output went back to the hosted model as part of the conversation. That output
included aggregate tables and also some raw material: individual raw tracking lines, a homography row, Smart
Tagging XML, public ground-truth rows, and several rendered video frames (viewed as images). Whether this is
compatible with the terms accepted at enrolment depends on those terms, which are not reproduced in the repository.
It cannot be undone. What can be done: ask the organisers, state the facts in the technical document (section 10 of
`docs/technical_document.md` has them, marked for the team to confirm), and avoid sending further raw data or video
frames to the hosted model.

## 2026-11-12 (v1b) - Pre-registration: a narrow change to the reference baseline (written before it was run)

**Aim.** Change only the parts of the reference baseline for which the tracking gives evidence, and keep its
smoothing, so the two changes are isolated from the segmentation choice that sank v1 (finding 46).

**Base.** The reference method: nearest tracked slot's team on each frame with a tracked ball, then a centred
majority vote over +/- 50 frames (ties to home). Where the window holds no vote the reference writes `D`.

**The two changes**

1. **Hold.** Where the window holds no vote (ball not tracked for the whole window), keep the previous
   vote instead of writing `D`: forward fill within each half, back fill at the head of a half. Reason:
   a missing ball is mostly live play (finding 32, and finding 35 for what a stoppage looks like).
2. **Dead ball from the detector.** `D` on the frames flagged by the stoppage detector (theta 1.0 m/s, T 5 s)
   and on the second-half frames 110808 to 113741 (`FAQ.md`: graded, before kick-off, all `D`), written
   over whatever the vote gave.

**Four variants, same smoothing, all reported with equal prominence**

| variant | hold | detector `D` |
|---|---|---|
| A: reference, window 50 | no | no |
| B: hold only | yes | no |
| C: detector only | no | yes |
| D: **v1b** = hold + detector | yes | yes |

**Reported:** the validator on v1b; possessions, mean duration and shares of H, A, D per variant; how many
graded frames each change relabels relative to A; the official scorer on the public 5 minutes as a
description of all four, never as a selector. Plausibility bounds as for v1 (mean possession 5-40 s, dead
share 5-45%); outside them v1b is investigated before use.

**Not claimed.** That v1b beats the reference. The full-match truth is unavailable; the reference has a
published full-match spellF1 of 0.2433 and v1b's is unknown. Known weaknesses stated in advance: holding
assumes that possession does not change unseen (untested); the detector covers long stoppages only
(finding 39), so short stoppages inside a missing-ball stretch will be labelled as live play by the hold.

### Result of v1b (run exactly as pre-registered; nothing was changed afterwards)

Code: `possession/v1b.py`. File: `possession/output/v1b/possession_v1b.csv` (variant D, local). Tables and
provenance: `docs/results/possession_v1b.md`. The validator accepts the file. Variant A reproduces the committed
reference window-50 labels on all 158,857 graded frames (0 differ).

| variant | possessions | mean duration | H | A | D |
|---|---|---|---|---|---|
| A: reference, window 50 | 594 | 7.2 s | 38.6% | 29.1% | 32.3% |
| B: hold only | 517 | 12.3 s | 56.0% | 44.0% | 0.0% |
| C: detector only | 559 | 7.2 s | 35.9% | 27.2% | 36.9% |
| D: hold + detector (v1b) | 526 | 9.6 s | 44.6% | 34.6% | 20.8% |

| # | Finding | Evidence |
|---|---|---|
| 47 | v1b relabels 32,927 of the 158,857 graded frames (20.7%) relative to the reference. The hold alone relabels 51,368 frames from `D` to `H` or `A`; the detector turns 7,265 live frames into `D` (and puts back into `D` many of the frames that the hold had filled) | `possession_v1b.md` section 3 |
| 48 | The hold removes every `D` that came from a missing ball (D share 32.3% to 0%) and lengthens possessions (7.2 to 12.3 s, 594 to 517); adding the detector brings the D share to 20.8% (the detector alone flags 18.9% of the graded frames; with the stated second-half pre-kick-off frames 20.8%) | same |
| 49 | v1b is inside the pre-registered plausibility bounds (mean possession 9.6 s; dead share 20.8%) | same |
| 50 | Public window (description only; nothing chosen from it): B and D are identical there because the detector flags no frame in that window (finding 41); both give spellF1 0.3610 and macroF1 0.5157, against 0.3256 and 0.6207 for A and C. The spellF1 difference is +0.035, which is less than the effect of one possession on a window with 17 (EVALUATION.md puts it at about 6 points); the macroF1 fall is the `D` class, which the hold-based variants never predict in that window | same, section 4 |

**What this does and does not show.** It shows that the two changes are implemented as designed, that they alter a
fifth of the graded frames, and that on the public window the hold keeps live possessions continuous where the
reference wrote `D` (finding 32) while losing the `D` class. It does **not** show that v1b is better on the full
match: the public window cannot decide that, and the detector's contribution is untested there (it changes
nothing in the window). Of the two changes the hold is the better supported (the reference's `D` on a missing ball
is wrong for 164 live frames in the window and the missing ball is mostly live play), the detector the weaker
(it passes a pre-registered proxy test on the tags but finds no stoppage in the window).

**For the team.** Three valid candidates exist: A (the reference, published full-match spellF1 0.2433),
B (hold only) and D (v1b). Submitting is a single choice with no feedback before the deadline (the last
response before 17 November is scored). The evidence available cannot rank them; the choice is a judgment about
risk, and should be made knowing that.

## 2026-11-12 (v1) - Pre-registration: assembling the first full submission (written before it was run)

**Components and parameters, fixed in advance**

1. **Ball tracked:** the team of the nearest tracked slot, as in the reference baseline (verified in
   `possession_baseline.md`).
2. **Dead ball:** the stoppage detector with the cell selected by its own pre-registered procedure,
   theta 1.0 m/s and T 5 s. In addition, the second-half frames 110808 to 113741 (graded, before
   kick-off) are `D`: `FAQ.md` states that all of them are.
3. **Possession state:** a two-state (home or away) Viterbi path with a switch penalty of **75 frames
   (3 s)**. Evidence: on a frame with the ball tracked, staying in a state that differs from the nearest
   team costs 1; frames without a tracked ball, and frames flagged dead, carry no evidence; switching
   is free on frames flagged dead (possession can change at a restart). So a switch needs 3 s more
   evidence than staying. The 3 s is the challenge text's own proxy for "controlled possession" ("a
   player keeps the ball for a few seconds", `CHALLENGE.md`). It is not tuned. It is run separately for
   each half.
4. **Ball missing in live play:** no evidence, so the state is held (a consequence of 3, not a separate rule).
5. **Frames outside the two halves:** `X` (never scored).

**What will be reported, and what will not be used**

* Validator result; possession count, mean duration and dead share, next to the reference variants; the
  switch-penalty sensitivity (25, 50, 75, 125, 200, 300 frames) as a description. None of these selects a
  parameter.
* The official scorer on the public 5 minutes, as a correctness check and a description. It selects nothing.
* Rough plausibility bounds, written now: mean possession between 5 s and 40 s, dead share between 5% and
  45% of graded frames. Outside them, v1 is investigated before use. The bounds are football common sense,
  not derived from this match's truth.

**Risk, stated now.** v1 has no full-match ground truth to be checked against. The reference baseline
(window 50) has a published full-match spellF1 of 0.2433; v1's is unknown and may be lower or higher. No claim
is made that v1 beats it. Which file to submit is a decision for the team given that uncertainty.

### Result of v1 (run exactly as pre-registered; nothing was changed afterwards)

Code: `possession/v1.py`. File: `possession/output/v1/possession_v1.csv` (local). Full tables and provenance:
`docs/results/possession_v1.md`. The validator accepts the file.

| # | Finding | Evidence |
|---|---|---|
| 44 | Structure over the graded frames: H 44.2%, A 35.0%, D 20.8%; 151 possessions of mean 33.3 s. The reference variants over the same frames: window 50 gives 594 possessions of mean 7.2 s, window 125 gives 398 of 12.4 s. v1 is inside the pre-registered plausibility bounds (mean possession 5-40 s, dead share 5-45%), but its mean duration is near the upper bound | `possession_v1.md` sections 1 and 2 |
| 45 | The structure depends strongly on the switch penalty: 312 possessions at 1 s, 193 at 2 s, 151 at 3 s, 114 at 5 s, 97 at 8 s, 80 at 12 s. The tracking alone gives no way to choose it | section 2 |
| 46 | On the public window (a correctness check and description, nothing chosen from it): v1 scores macroF1 0.5255 and spellF1 0.1913, below the reference windows 25, 50 and 125 (spellF1 0.2868, 0.3256, 0.4242) and above the unsmoothed one (0.0378). v1 makes 6 possessions of mean 50 s there against 17 true ones of mean 14.2 s: it under-segments. Its macroF1 is also lowered by the dead-ball class, because the detector calls no frame dead in that window (finding 41) | section 3 |

**Decision taken: nothing was tuned after seeing finding 46.** Choosing the switch penalty from the public
window would be tuning on 17 possessions, which `EVALUATION.md` shows picks the wrong parameter, and `RULES.md`
forbids using the window as a tuning set. v1 as pre-registered therefore stands as the result: a version built
from reasoned components that, on the only truth available, under-segments and scores below the organisers'
reference. Its full-match score is unknown.

**Consequences and options (for the team)**

* The reference baseline (window 50) has a published full-match spellF1 of 0.2433. v1 is not shown to beat
  it and the public-window check points the other way. Submitting v1 as it is would be a bet without evidence.
* A narrower change keeps the reference's smoothing and changes only what the tracking evidence supports: dead
  ball from the stoppage detector, and holding the label through a missing ball instead of writing `D`. That
  isolates those two changes from the segmentation choice.
* A way to choose the switch penalty without ground truth (for example, how well the path explains the
  nearest-team evidence as the penalty grows) could be pre-registered as a v2, with the disclosure that it was
  designed after finding 46.
* Using the published full-match scores of the reference windows to infer how long possessions are would use
  information derived from the private ground truth and is not done here.

## 2026-11-12 (dead ball) - What stoppages look like in the tracking, and the pre-registered detector test

**What was looked at first** (descriptions of the whole match from tracking and Smart Tagging only; no
ground truth; nothing was tuned on it)

| # | Finding | Evidence |
|---|---|---|
| 35 | Before restarts the ball is mostly not tracked: in the 20 s before the tag start it is tracked on 11% of frames for corners, 10% for free kicks, 11% for kick-offs after a goal, 44% for goal kicks and 56% for throw-ins (match-wide: 59%) | ad hoc exploration over the 59 `SET PIECES` tags |
| 36 | A ball that is tracked and still for 3 s or more is rare: 15 such runs in the whole match, and only 12% of set pieces have one close by. So "ball stationary" cannot be the general dead-ball signal | same |
| 37 | The players' rolling (3 s) median speed dips before long restarts: corners 1.5 to 0.64 m/s, free kicks 1.6 to 0.46, kick-offs after a goal about 0.26 to 0.9, then jumps to 2 m/s or more 10 s after the tag start. Goal kicks and throw-ins show no dip (they restart fast). Match-wide the rolling speed has median 1.4 m/s and 10th percentile 0.62 m/s | same |
| 38 | The ball-tracked share falls to 31% at the restart (59% match-wide) and recovers to 81% ten seconds later | same |

These are not saved as a script yet (ad hoc); the detector report below will recompute what it uses.

**Question.** Can a simple rule from the tracking alone find the long stoppages (the dead-ball stretches
that end in corners, free kicks, kick-offs and breaks)? The reference baseline's rule, "ball not tracked
means `D`", misses most dead ball in the public window and wrongly marks live play (findings 31, 32).

**Rule (no learning).** A frame is a dead-ball candidate when s(t) < theta, where s(t) is the 3 s rolling
mean of the median speed of the tracked slots (speed column of the tracking), AND the ball is not
moving: not tracked, or displaced less than 1 m over a centred 2 s window. A stoppage is a maximal run
of candidates lasting at least T seconds; runs separated by less than 2 s are merged. Grid:
theta in {0.5, 0.6, 0.8, 1.0} m/s, T in {3, 5, 8} s.

**Evidence available.** The 59 Smart Tagging `SET PIECES` tags (automatic, not human labels): corner 12,
free kick 5, kick-off 6 (the "long restarts", 23 together) and throw-in 26, goal kick 10 (the "short
restarts"). The tags are incomplete (a real match has more throw-ins than 26), so a "precision"
measured against them is pessimistic.

**Metrics.**

* Recall proxy R: share of long-restart tags (corner, free kick, kick-off) with a detected stoppage that
  ends in [tag start - 3 s, tag start + 10 s]. Also reported for throw-ins and goal kicks.
* Precision proxy P: share of detected stoppages that end in that window around any set-piece tag.
* Also reported: dead share of graded frames, median stoppage length.

**Selection rule (written before the run).** Choose the grid cell that maximises the harmonic mean of
R (long restarts) and P; ties go to the larger T, then the lower theta. The public ground truth is not
used for selection. It is shown afterwards as a description of the chosen rule (not a score to improve).

**Failure criterion.** If no grid cell has R of at least 0.5 and P of at least 0.3, the rule is
declared not usable and the next step is a probabilistic two-state model (live or dead) fitted without
labels, instead of a threshold.

**Caveats written in advance.** 23 long-restart tags is a small sample; the grid is reported in full so
sensitivity is visible. The rule is not expected to find throw-in or goal-kick stoppages.

### Result of the pre-registered test

Code: `possession/deadball.py`, `possession/deadball_report.py`. Full grid, by-type table and public-window
description with provenance: `docs/results/deadball_detector.md`. The procedure above was followed unchanged.

**Under the failure criterion the rule is usable** (cells with theta of 0.8 m/s or more pass R >= 0.5 and
P >= 0.3). Selected by the rule: **theta 1.0 m/s, T 5 s**: R long 0.70, R short 0.39, P 0.45, F1 0.55, 64
stoppages, 19% of graded frames dead, median stoppage length 12.2 s.

| # | Finding | Evidence |
|---|---|---|
| 39 | By restart type (stoppage ends within [-3 s, +10 s] of the tag start): corner 92%, kick-off after a goal 75%, throw-in 42%, free kick 40%, goal kick 30%, opening kick-off 0%. The rule finds the long stoppages and misses most short ones | `deadball_detector.md` section 2 |
| 40 | The selected cell is at the edge of the grid (theta 1.0 is the largest value) and F1 was still rising with theta (T 5 s: 0.22, 0.33, 0.47, 0.55). The best value may lie beyond the grid. It was not extended after seeing the results, because that would be a post hoc change to the pre-registered search | grid table |
| 41 | **On the public window the rule calls no frame dead**, and finds none of the 4 ground-truth dead-ball stretches. The proxy test on the tags and the public window therefore disagree about what stoppages the rule covers | `deadball_detector.md` section 3 |
| 42 | Why (ad hoc diagnosis, public window): inside the 4 dead-ball stretches the players' rolling speed is 1.0 to 1.95 m/s (median 1.16, 1.01, 1.20, 1.95) while live frames of this window have median 2.18; in 2 of the 4 the ball is tracked and moving (median 2.7 and 1.6 m/s over 1 s), starting near or outside a touchline (one at x = -1.4 m), i.e. it is carried back to a restart. One stretch ends 2.7 s after a goal-kick tag start and one 12 s after a free-kick tag start; the other two have no set-piece tag, so the tags are incomplete (the window has only 2 restart tags, the goal kick and the free kick, for its 4 stoppages; its third set-piece tag is the opening kick-off) | ad hoc diagnosis, `explore` data; tags |
| 43 | Ball tracked outside the pitch (margin 0.5 m) is rare, 2.1% of graded frames in 71 runs of median 0.9 s, but it appears in the 30 s before 13 of 26 throw-in tags and 5 of 10 goal-kick tags (2 of 12 corners). A partial signal for the start of short stoppages | ad hoc exploration, tags only |

**Reading.** From the tracking alone, long stoppages (corners, kick-offs after goals, breaks, some free kicks)
can be found with a simple rule; short ones (throw-ins, goal kicks) mostly cannot, because players keep
moving and the ball is often carried or untracked. Which kind dominates the dead-ball time of a match is not
known: the public window suggests short stoppages are common, and the Smart Tagging marks fewer restarts
than a match has. The proxy evidence (tags) is biased towards the stoppages the tags cover, so "usable"
means "finds a real part of the dead ball", not "finds dead ball".

**What was NOT done:** the rule was not evaluated for its effect on the possession score, and the public
window was not used to tune it. A union with a ball-outside-the-pitch signal was looked at only
descriptively (finding 43).

## 2026-11-12 (baseline) - Our own end-to-end possession baseline, reproduced and reviewed

**Done**

* `possession/` (`tracking.py`, `baseline.py`, `report.py`, `README.md`): tracking to per-frame label to
  CSV to validator to official scorer, using the reference method (nearest player to the ball, majority
  vote over a window), written from scratch. Results with provenance:
  `docs/results/possession_baseline.md`. Run: `python possession/report.py`. Needs numpy and pandas only.
* Decision: the method is the reference baseline on purpose. It is the floor and the pipeline everything
  else plugs into; building the pipeline first meant checking our reading of the data against a known
  answer before adding anything.

**Findings**

| # | Finding | Evidence |
|---|---|---|
| 30 | Our labels are identical to the committed reference baselines on all 199,319 frames for all four variants (windows 0, 25, 50, 125), and the official scorer gives the published public-window scores (smooth125 0.5812 / 0.4242, smooth50 0.6207 / 0.3256, smooth25 0.6200 / 0.2868, raw 0.5972 / 0.0378 macroF1 / spellF1) | `possession_baseline.md` sections 1 and 2 |
| 31 | Ground-truth dead ball is often tracked: the ball is tracked on 66.4% of the `D` frames of the public window, and on 100% of two of the four dead-ball stretches (20.4 s and 14.5 s). The reference rule "ball not tracked means `D`" labels only 314 of the 1,469 `D` frames `D` (902 become `H`, 253 `A`) | sections 3.1 to 3.3 |
| 32 | A missing ball is often live play: the ball gaps of 5.3 s and 9.3 s lie entirely in ground-truth `A`; 164 `A` frames are labelled `D` by smooth50 | sections 3.2 and 3.4 |
| 33 | The raw nearest-team rule is right on 77.8% of live frames with the ball tracked, falling with distance: 90.6% (nearest player under 1 m), 82.6% (1-2 m), 68.8% (2-4 m), 62.0% (4-8 m) | section 3.5 |
| 34 | smooth50 yields 26 possessions in the public window against 17 true ones; over the whole match the validator reports 669 possessions of mean 7.6 s for smooth50 | scorer board; `validate_submission.py` |

**Caveat.** Findings 31 to 33 come from the public window: 7,500 frames, 17 possessions, 4 dead-ball
stretches and 4 ball gaps of 1 s or more. They are leads to test on the whole match with tracking-only
evidence, not rules. The public ground truth was used only to describe the baseline's errors; no
threshold, rule or model was chosen from it.

**What this suggests (hypotheses)**

* Dead ball needs its own detector instead of "the ball is missing": for example, how long the ball has
  been nearly stationary, how the players are spread, and the Smart Tagging set pieces (restarts).
* A missing ball in live play should keep the previous possession (hold the label) instead of writing `D`
  (in a submission an `X` is forward-filled by the scorer for `spellF1`).
* Per-frame decisions need segmentation (a possession has a beginning and an end, with a switch penalty)
  because the possession count is too high.

**Next**

1. Test on the whole match, with tracking and Smart Tagging only: how often the ball is tracked but
   stationary for long stretches, and how those stretches line up with set pieces.
2. Hold-the-label policy for missing ball outside detected dead-ball stretches.
3. A segmentation step with one switch-penalty parameter.
4. Only then a boosted-tree per-frame classifier for frames where the ball is not tracked, validated by
   time-ordered splits (no ground truth for the full match exists to tune against).

## 2026-11-12 (later) - Pre-registration: residual-on-interpolation variant (written before the run)

This entry is committed before the runs start, so that the rule for judging them cannot be adjusted
after the numbers are known.

**Question.** Finding 21 says the first model ignores the tracked ball at the gap edges. If the model
outputs a *correction added to linear interpolation* (and receives the interpolation as input), can the
players add information beyond what interpolation already gives?

**Protocol.**

* Split v2 (`recovery/data.py`): train = half 1 minutes 0-36 + half 2 minutes 0-20; validation = half 2
  minutes 21-31; `test_seen` = half 2 minutes 32-end (already looked at once, flagged in every table);
  **`test` = half 1 minutes 38-end, clean**: never used for any decision, scored once per model.
* Two models, same split, seeds, 50 epochs, patience 10, 40 batches of 16 per epoch:
  (A) direct output, the first architecture trained longer; (B) residual on interpolation. A is the
  ablation: the difference between A and B is the architecture alone.
* Baselines (last position, constant velocity, linear interpolation) on the same trials.
* The epoch is chosen by mean validation error over the four gap lengths. Nothing else may use `test`.

**Decision rule.**

* B counts as better than linear interpolation at a gap length only if its paired difference on the
  clean `test` has a 95% bootstrap interval entirely below 0 at that length. All four lengths are
  reported; with four comparisons, one marginal win is not a finding.
* If B is not better at any length, the conclusion is that the players, as used by this model and
  this amount of data, add no recoverable information beyond interpolation, and linear interpolation
  is the recovery method to use.
* If model code, split or hyperparameters change after the clean-test numbers are seen, `test`
  is no longer clean and a new held-out block is required before any new score is called untouched.

**Not part of this run:** the past-only (`--causal`) variant, evaluation on gaps shaped like real
ones, and any use of the recovered ball for possession.

## 2026-11-12 (result) - Residual-on-interpolation: no gain over linear interpolation

Both runs followed the pre-registration above, unchanged (commit `452761c` precedes them). Full tables:
`docs/results/ball_recovery_bilstm_residual_split2.md` and `..._direct_split2.md`.

**Clean test (first half minutes 38-end), paired difference vs. linear interpolation in metres, 95% bootstrap interval** (negative = better; 53 / 43 / 24 / 14 trials)

| gap | residual on interpolation | direct (trained longer) |
|---|---|---|
| 1 s | +1.07 [+0.88, +1.24] | +11.80 [+9.82, +14.04] |
| 2 s | +0.26 [-0.03, +0.57] | +10.32 [+7.68, +13.34] |
| 5 s | +0.32 [-0.16, +0.74] | +7.77 [+3.81, +12.02] |
| 10 s | -0.31 [-0.82, +0.24] | +1.28 [-3.46, +6.53] |

**Decision under the pre-registered rule:** the residual model is not better than linear interpolation
at any gap length (no interval lies entirely below 0). So, for this data and this model family, the
players add no recoverable information beyond interpolation for hidden stretches of up to 10 s, and
linear interpolation is the recovery method to carry forward for short gaps.

| # | Finding | Evidence |
|---|---|---|
| 24 | The residual model is slightly worse than interpolation at 1 s (+1.07 m, interval above 0) and indistinguishable at 2, 5 and 10 s | table above |
| 25 | Its best validation epoch was epoch 1 (5.84 m); validation error then rose (7.3, 13.4, 11.8, 10.5, 9.6, 10.8, 7.6, 6.9, 8.6, 9.4 m) and training stopped at epoch 11. Interpolation alone scores about 5.1 m on the same validation trials (mean of 0.81, 2.02, 5.78, 11.91 m). Training moved the model away from interpolation and never got back | `recovery/output/training_log_bilstm_residual_split2.csv`, validation table |
| 26 | The direct model, trained longer (32 epochs, best epoch 22, validation 8.49 m), is still worse than interpolation up to 5 s on all three periods (clean test +11.8, +10.3, +7.8 m at 1, 2, 5 s) | results file |
| 27 | At 10 s the direct model showed a marginal win on validation (-3.21 m [-6.51, -0.11]) that did not appear on the clean test (+1.28 [-3.46, +6.53]) or the earlier test (+1.82 [-1.67, +5.79]). This is why the rule required the clean test | results files |
| 28 | Team agreement: residual 92.1 / 88.8 / 77.7 / 54.0% at 1 / 2 / 5 / 10 s, interpolation 95.1 / 89.8 / 77.8 / 56.0%; the direct model is about chance (52-60%) | results file |
| 29 | The clean test has only 134 trials, so intervals at 5 s and 10 s are wide. "No detectable difference" is not proof of none; the size of any gain at 10 s is under about 1 m against an error of about 13 m | table above |

**What this means for the project (hypotheses, not yet tested on possession)**

* Short gaps (up to a few seconds) can be filled by linear interpolation; nothing tried here does better.
* Long gaps cannot be recovered well by any method tried: at 10 s the methods that use the ball on both
  sides of the gap (interpolation, residual, direct) are at 11-14 m, and the past-only baselines at
  22-24 m. Gaps over 30 s hold 40-48% of the missing frames (finding 16), so most of the missing ball is
  out of reach.
* So possession labels in long gaps have to come from the players' collective behaviour, restarts and
  stoppages (findings 15 and 17), not from a reconstructed ball path. Whether that works is the next
  question.

**Not tested:** the past-only (`--causal`) variant, gaps longer than 10 s, more training data, other
model families, gaps shaped like real ones, any effect on possession labels. The model is an adaptation
of the paper's, not a replication (see `recovery/README.md`), and it was trained on one match.

## 2026-11-12 - Ball recovery with a Set Transformer + hierarchical Bi-LSTM: it does not beat linear interpolation

**Done**

* Branch `explore/ball-recovery`. Added `recovery/` (`data.py`, `baselines.py`, `model.py`, `run.py`,
  `README.md`): hide stretches of tracked ball (1, 2, 5, 10 s), reconstruct them, score against what
  was hidden. Time-ordered split, fixed trials, bootstrap confidence intervals. Model adapted from
  Kim et al., KDD 2023 (arXiv:2306.08206); differences from the paper are listed in
  `recovery/README.md` (algorithmic possession pseudo-labels instead of event data, no reality loss
  because its printed sign is unresolved, about 1M parameters).
* Full results with provenance: `docs/results/ball_recovery_bilstm.md`. Command:
  `.venv/bin/python recovery/run.py` (seed 0; evaluation trials seed 12345).

**Result (test period, half 2 minutes 32-end, mean position error in metres; trials 103 / 84 / 53 / 27)**

| gap | last position | constant velocity | linear interpolation | Set Transformer + Bi-LSTM |
|---|---|---|---|---|
| 1 s | 4.00 | 1.80 | **0.77** | 10.80 |
| 2 s | 7.01 | 4.36 | **2.17** | 11.49 |
| 5 s | 12.69 | 13.11 | **6.33** | 13.30 |
| 10 s | 23.83 | 21.75 | **11.86** | 14.87 |

| # | Finding | Evidence |
|---|---|---|
| 18 | The learned model is worse than linear interpolation at 1, 2 and 5 s (paired differences +10.0, +9.3, +7.0 m, confidence intervals all above 0). At 10 s it is not distinguishable: +3.0 m [-0.66, +7.28] on test and -1.94 m [-5.21, +1.39] on validation | results file, paired tables |
| 19 | Linear interpolation is the strongest method for gaps up to 5 s. Constant velocity beats holding the last position up to 2 s and is not clearly better at 5 and 10 s (confidence intervals overlap) | results file |
| 20 | Nothing recovers a 10 s gap well: the best error is 11.9 m (linear interpolation) | results file |
| 21 | The model's error barely depends on gap length (10.8 m at 1 s, 14.9 m at 10 s), while the interpolation error grows from 0.8 to 11.9 m. A 10.8 m error at a 1 s gap means the model is not using the tracked ball next to the gap at all: it estimates "where the ball tends to be given the players" | results file |
| 22 | Its team agreement (proxy: team of the player nearest to the guess vs. to the true ball) is 45-59%, about chance, against 94% for linear interpolation at 1 s | results file |
| 23 | The best validation epoch was the last one (25 of 25) and the validation error was still falling (9.20 m), so training was too short | `recovery/output/training_log_bilstm.csv` |

**Reading of the result**

This is a negative result for the model as built, in this budget: one match, about 73 minutes of
training data, 25 epochs. It is not evidence against the approach in general. The likely causes, not
yet tested, are (a) too little training, (b) no direct path from the tracked ball at the gap edges to
the output, so the network has to learn to copy it through two LSTMs, (c) very little data for a
model of this kind (the paper used a dozen matches). The natural next model predicts a correction on
top of linear interpolation, so it can only add information from the players.

**Test-set hygiene.** The test period was scored once, and has now been seen. Further model changes
must be chosen on the validation period, and a model chosen after looking at these test numbers needs
a fresh held-out block (for example the last 15 minutes of half 1, taken out of training) before its
test score is reported as untouched.

**Next**

* Residual-on-interpolation variant and longer training, judged on validation only; new untouched
  test block.
* The past-only variant (`--causal`) for the stricter comparison.
* Evaluate on gaps that look like real ones (finding 15: fewer tracked players, near the goals, slow
  ball), and handle stoppages (finding 17) as dead ball rather than as gaps.

## 2026-11-11 (later) - Missingness on the scored frames, and constraints for self-supervised recovery

**Done**

* Added `explore/missingness_summary.py`. It reads only the tracking (no ground truth) and writes
  `docs/results/missingness_summary.md` (tables + provenance: command, time, git commit, script and
  input SHA-256, library versions), plus a per-minute CSV and a picture in the git-ignored
  `explore/output/`. Run: `python explore/missingness_summary.py`.
* This replaces the ad hoc computation behind findings 8 and the slot-count part of 9 below. The
  location and speed parts of finding 9 are still ad hoc.

**Findings** (scored frames only: the two halves, 158,857 frames; full tables in
`docs/results/missingness_summary.md`)

| # | Finding | Source |
|---|---|---|
| 13 | Ball missing on 40.7% of frames in each half (41.6% in the second half counted from kick-off). Mean tracked slots 20.1 (half 1) and 20.9 (half 2) of 46, about 10 per team | script, table 1 |
| 14 | Minute by minute the ball-missing rate ranges from 0% to 100% (median 39%). Of 104 full minutes, 19 are under 10% missing and 15 are over 75%. Three minutes are 100% missing: half 1 minute 32, half 2 minutes 32 and 33 | script, table 3 |
| 15 | Tracked slots per frame range from 1 to 25. Half the frames have exactly 22 and a quarter 21; 7.7% have 16 or fewer. Ball missing climbs with fewer tracked slots: 26.6% at 22 slots, 55.4% at 20, 88.9% at 16 or fewer. Per-minute correlation between ball-missing % and mean tracked slots is -0.56 | script table 2 + per-minute CSV |
| 16 | Ball gaps: 69 in half 1 (median 11.6 s, p90 43.3 s, max 138.6 s) and 81 in half 2 (median 9.8 s, p90 25.6 s, max 221.5 s). Gaps of 1 s or less are 17 of 150. Gaps over 30 s hold 47.9% (half 1) and 39.8% (half 2) of the missing frames. Uninterrupted tracked runs have a median of 15.2 s and 12.0 s | script, table 4 |
| 17 | The 100%-missing minutes at minute 32 of both halves are not tracker failures in the usual sense: the video shows players and staff gathered at the touchline with the match clock running (32:28 and 75:31), so the ball is out of play; only 7 to 13 slots are tracked. Looks like a drinks/cooling break. Seen in frames 59935 and 159558; the label for these frames is not public, so "dead ball" is a hypothesis, not a result | `frame_explorer.py --frame 59935` and `--frame 159558`, visual inspection |

**What this means**

* Gaps last seconds, not frames, so extrapolating the ball from its last position, speed and angle
  is only useful for the first second or two of a gap. Most missing frames need the players' collective
  behaviour and the structure of the match instead.
* Missingness is clearly not MCAR: it depends on observed variables (how many slots are tracked, where
  the ball was last seen, how fast it was moving) and so is at least MAR. Whether it also depends on the
  hidden ball state (MNAR) cannot be tested from the tracking alone; finding 17 is one concrete case
  where it does (no ball because play is stopped). That has to be argued and probed with sensitivity
  analysis, not "classified".
* Breaks and other stoppages are missing-by-design. They should be modelled as dead ball, not as gaps
  to fill.

**Design constraints for self-supervised ball recovery** (decided before any model is written)

1. **Split by time blocks, never by random frames.** Frames 1/25 s apart are near duplicates, so a
   random split leaks. Train and validate on separate time blocks with a buffer between them at
   least as long as the model's context window; fit any normalisation on training blocks only.
2. **Mask contiguous gaps that look like the real ones.** Draw lengths from the observed distribution
   (finding 16), not single frames, and place them where real gaps occur (slow ball, fewer tracked
   slots), because missingness is not random (finding 15).
3. **Match the context the real gaps have.** The submission is produced offline, so using frames
   before and after a gap is allowed (`RULES.md` only requires that an algorithm produce the labels).
   Real gaps have tracked ball on both sides (finding 16, tracked runs), so validate with two-sided
   context and also report a past-only variant as a stricter baseline.
4. **Do not train on gaps that cannot be recovered.** Breaks (finding 17) have no ball to recover;
   keep them out of the "recoverable" validation set and handle them as dead ball.
5. **Coordinates are absolute and teams swap ends** (finding 5): normalise by attacking direction per
   half, or give the half as an input, and test transfer from half 1 to half 2.
6. **Judge recovery by possession, not only by position error.** The ranking metric is `spellF1`;
   a recovered ball path that is 5 m off but gives the right possession is better than the reverse.
7. **The public 5 minutes are for pipeline checks only**, never for choosing thresholds or models.
8. **Provenance for every experiment:** seed, the exact split (block boundaries), command, git
   commit and input hashes, written next to the result, as `missingness_summary.py` does.

**Next**

* Measurement error of the ball track: implausible frame-to-frame jumps and accelerations, and a
  comparison with the independent (unsynchronised) broadcast ball track.
* Test MAR formally: regress the missingness indicator on slots tracked, last-known ball location and
  speed, with time-blocked cross-validation.
* Check how many gaps coincide with Smart Tagging set pieces (restarts) and with breaks.

## 2026-11-11 - Frame explorer and first look at the data

**Done**

* Added `explore/frame_explorer.py`, `explore/01_frame_explorer.ipynb` and `explore/README.md`: show
  one video frame with its tracking (projected through the homography), pitch diagram, ball status,
  homography quality, Smart Tagging phases and the public label. See `explore/README.md`.
* Added `explore/output/` to `.gitignore` (rendered frames are derived from gated data).

**Findings** (all on the tactical recording)

| # | Finding | How obtained |
|---|---|---|
| 1 | Video: h264, 1920x1080, constant 25 fps, 199,319 frames, starts at t = 0 | `ffprobe` |
| 2 | Our video file is `BAR-ATH_tactical-002.mp4`, not the name `games.py` expects | `ls data/tactical`; handled in `find_video()` |
| 3 | `ffmpeg -ss frame/25` is frame-accurate: pixel-identical to a sequential decode at frames 11500 and 20000 | `python explore/frame_explorer.py --check-seek` |
| 4 | Projected tracking lands on the real players in the video (frames 11500, 135406); the burned-in clock reads 00:11 at frame 11500 | visual inspection of the rendered figures |
| 5 | Coordinates are absolute and teams swap ends at half-time: slot 24 mean x = 0.10 (half 1), 0.88 (half 2); mean x of slots 1-24 / 25-46 is 0.47 / 0.58 in half 1 and 0.53 / 0.41 in half 2 | **ad hoc** script, every 25th frame of each half |
| 6 | Slot 24 is Barcelona's goalkeeper in most frames, but slots swap identity: at frame 135406 it is at the left goal while Barcelona defend the right | frame 11500 and 135406 figures + finding 5 |
| 7 | "Frozen" slots (position identical to previous frame) are rare: 0-0.2% for most slots, 1.3% for slot 24, 2.4% for slot 46 (slot 10 also 2.4% but from only 83 frames) | **ad hoc** script, first half, every 3rd frame |
| 8 | Ball missing on 40.7% of the 158,857 graded frames, in 150 gaps with median length 268 frames (about 10.7 s). 99.7% of missing frames lie in gaps longer than 5 s; there are only 14 gaps of 5 frames or fewer | **ad hoc** script on the tracking only (no ground truth) |
| 9 | Missingness is not random: ball missing on 66% / 57% of frames when its last known position is near either goal end (x < 0.17 / x > 0.83) vs 28% in mid-pitch; 87% when 17 or fewer slots are tracked vs 28% when 22 or more; gaps start when the ball is slow (median 2.1 m/s before a gap vs 5.1 m/s overall) | same **ad hoc** script |

**Raw view added (`--raw`, `fe.format_raw`)** - requested so the data can be understood as it
arrives. For frame 11500 it prints the tracking line (1,010 characters, 21 of 46 slots tracked), the
homography row with a worked pixel computation, the metadata XML per slot, the five Smart Tagging
`<instance>` blocks covering the frame, and the ground-truth rows. Verified byte-for-byte against
the files with `sed` (tracking line 11501, homography line 11502, ground-truth row 11500 = `H`).

| # | Finding | How obtained |
|---|---|---|
| 10 | The homography CSV and the Smart Tagging XML use `\r\n` line endings; the tracking text and the ground-truth CSV use `\n` (first 2,000 lines of each checked). A first byte comparison of the homography row failed only because of the trailing `\r`; content was identical | `od -c`, `grep -c`, `sed` |
| 11 | Worked example on one frame: the ball position, mapped through that frame's homography, lands on the ball in the video to within a pixel, and the ring is drawn there | `python explore/frame_explorer.py --frame <frame> --raw` (local only) |
| 12 | Smart Tagging phases overlapping one frame can name both teams (four name one team and one the other, on a frame whose public label is the latter). Phases are not possession labels | same command |

**Implication (not yet tested):** because gaps are long (seconds), short-gap interpolation will not
recover them. The missingness itself, and the players' collective positions during a gap, are the
information to use. Whether gaps coincide with dead-ball restarts (Smart Tagging set pieces) is the
next check.

**Corrections to earlier statements**

* The project document and an earlier chat message said slot 24 sits at a fixed point in every
  frame and looks like a dummy track. That was wrong: it is a real player (usually the goalkeeper).
  The document's pitch chart now includes it, and the sentence has been replaced.

**Open / next**

* Move the ad hoc scripts (findings 5, 7, 8, 9) into the repository so each number is reproducible
  with one command.
* Overlay Smart Tagging set pieces on the ball-gap timeline and check how many gaps are dead balls.
* Look at frames inside long gaps to see whether the ball is visible in the video (decides whether
  our own detector could recover it).
