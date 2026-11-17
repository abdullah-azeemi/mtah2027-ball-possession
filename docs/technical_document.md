# Technical document: More than a Hack 2027, Phase 1 (ball possession)

**Team:** Prompt-ly &amp; Furious &nbsp;·&nbsp; **Members:** Abdullah Musharaf, Muhammad Ahmad Shahzad &nbsp;·&nbsp; **Repository:** https://github.com/abdullah-azeemi/mtah2027-ball-possession &nbsp;·&nbsp; **Version of:** 2026-11-17

## 1. Summary

We label every frame `H`, `A`, `D` or `X` with a small pipeline that uses only the tactical tracking and has no
learning. A frame is evidence of possession only when **exactly one team has a tracked player within 1.5 m of the
ball** and the ball is not in flight (slower than 7 m/s) (controlled possession); a centred majority vote over +/-60 frames turns that evidence into possessions, and
contested, loose or untracked-ball frames keep the previous possession, which is the challenge's own rule. We write
`D` on **long stoppages** found from the players' collective speed, on **untracked-ball runs of 10 s or more**, and
**while the ball is out of play** (tracked beyond a touchline or goal line, until the restart kick).

Honest status: we reproduced the organisers' baseline exactly (0 of 199,319 frames differ), measured what makes the
data hard, and tried a neural ball-recovery model and a Viterbi segmentation, both of which we dropped because the
evidence did not support them. The current candidate (`possession/submit.py`) keeps T_E's dead-ball rules unchanged
and changes only the per-frame team evidence to controlled possession. That change was tested, pre-registered, on two
other matches (Metrica's open sample games, event-derived possession): spellF1 0.521 -> 0.575 and 0.488 -> 0.635
against T_E's nearest-player evidence (section 7). On the public 5-minute check it scores spellF1 **0.5455** and
macroF1 **0.7766** with 16 predicted possessions against 17 true, versus T_E's 0.3905 and 0.7567 with 25. That window is a correctness check, not a tuning set, and with 17 possessions it cannot
rank two files reliably: **we do not know the full-match score of any candidate.** The sections below say what is
measured, what is reasoned and what is untested.

## 2. Approach and why

1. **Evidence per frame.** A frame casts a vote for a team only when **exactly one team** has a tracked player
   within 1.5 m of the ball (distances in metres): that is controlled possession. If both teams are within range
   (a contest), nobody is (a loose ball), the ball is untracked, or the ball is **in flight** (centred speed over
   0.16 s above 7 m/s: a pass or clearance, not a ball at someone's feet; our tracking has no ball height, so a
   ball flying past or over a defender would otherwise count as his), the frame casts no vote.
2. **Smoothing.** A centred majority vote over +/- 60 frames of the gated votes. A possession is a run of one team,
   so a touch, a deflection or an aerial duel shorter than the window does not change possession. The window is
   wider than the reference's 50 because gating removes about half of the frames from the vote; +/-60 gated frames
   carry about the evidence of +/-50 ungated ones.
3. **Hold.** Where the window holds no vote, the previous vote is kept (forward fill inside each half); a tie in the
   vote also keeps the previous decided possession instead of going to home. **Long gaps.** Runs of 250 or more
   frames (10 s) with the ball untracked are written `D` instead (see section 7).
4. **Dead ball.** `D` on stoppages (section 5) and on the second-half frames 110808 to 113741, which `FAQ.md` states
   are graded and all `D`.
5. **Ball out of play.** `D` from the moment the ball is tracked more than 0.5 m beyond a touchline or goal line for at
   least 0.4 s, until the restart: the first moment, at least 2 s later, when the ball is back on the pitch and moving at
   6 m/s or more (a throw or a kick; a ball being carried back is slower). A stretch ends after 60 s at most. This
   catches throw-ins, corners and goal kicks while the ball is still visible (section 7).
6. **Outside the two halves:** `X` (never scored).

**Why this and not something more elaborate.** There is no full-match ground truth, and the 5 public minutes may not
be used for tuning. A learned or tuned component therefore cannot be validated, and the organisers' own baselines show
that tuning on the public window picks the wrong parameter (`EVALUATION.md`). We chose to change only what tracking
evidence supports, to write the design, parameters and decision rule down **before** each run (`docs/worklog.md`), and
to report negative results. The controlled-ball rule is not tuned: it is the challenge's own definition
(`CHALLENGE.md`, "Ball in play": "In phases where the ball is contested or loose, possession stays with the last team
that had it"), applied at frame level. The reference counted a frame as evidence for the nearest team however far
away that player was, which is not control. We did compare T_E and the controlled-ball candidate on the public
window before choosing; section 7 reports that comparison and its limits, and the chosen parameters sit on a broad
plateau rather than at a tuned maximum.

## 3. Assets used and dropped

| Asset | Use |
|---|---|
| Tracking: positions and ball | used (steps 1 to 3) |
| Tracking: speed column | used (stoppage detector) |
| Tracking metadata XML | used (team of each slot; first team listed is home) |
| Smart Tagging | used only to **evaluate** the stoppage detector (restart tags); not in the labels |
| Homography and tactical video | used to inspect the data (`explore/`: tracking projected onto frames); not in the labels |
| Broadcast video, tracking, homography | not used (not aligned to the tactical timeline; we did not try to map it) |
| Public 5 minutes of ground truth | used as a correctness check, to **describe** where baselines err, and to compare the T_E and controlled-ball candidates (section 7). That comparison informed the candidate choice, which is a deviation from using the window only as a check; we disclose it and did not tune thresholds to its maximum |

## 4. Training data

None for the submitted file: it contains no learned component.

The only labelled data used anywhere is **Metrica Sports' open sample data** (`github.com/metrica-sports/sample-data`):
three anonymised matches of other teams, with tracking and event data produced by Metrica Sports (games 1 and 2:
1,745 and 1,935 events; game 3 in the EPTS format). Other games are allowed by `RULES.md`. It was used to:

* **test** the evidence rule of the submitted file, with possession built from the events of games 1 and 2
  (section 7, `docs/results/evidence_layer.md`);
* **train and test** the gradient boosting and random forest models (games 1 and 2, one held out; not submitted);
* **train** candidate G, a per-gap logistic regression (not submitted), and **test** the out-of-play rule (all three).

Experiments that learned on our match (`recovery/`) were self-supervised on its automatically tracked ball, with
targets produced by an algorithm (nearest player within 3 m of the tracked ball). **About the provided match:** no
hand labels, no event data or other human-made data, and nothing from the private ground truth, anywhere.

## 5. Possession rules implemented

* **Possession** is controlled possession: a frame votes for a team only when exactly one team has a tracked player
  within 1.5 m of the ball, and a centred vote over +/-60 frames decides. Loose and contested balls cast no vote and
  keep the last possession until the other team's controlled evidence wins a window.
* **Untracked ball:** for runs shorter than 10 s the previous possession is held (assumes possession does not change
  unseen); for runs of 10 s or more `D` is written, because a manual look at 10 long gaps found dead ball in all of them.
* **Dead ball:** a stoppage is a run of at least 5 s in which the 3 s rolling mean of the median speed of the tracked
  slots is below 1.0 m/s **and** the ball is not moving (not tracked, or displaced under 1 m over 2 s); runs less than
  2 s apart are merged.
* **Ball out of play:** the ball wholly over a line is out (Laws of the Game, law 9). Tracked more than 0.5 m beyond a
  line for 0.4 s (a guard against tracking noise) starts a dead stretch; a throw or kick at 6 m/s or more back on the
  pitch, at least 2 s later, ends it; 60 s at most.
* **Restarts** are otherwise not modelled; the possession after a stoppage comes from the vote.
* **Not handled:** short stoppages where the ball stays on the pitch (most free kicks after fouls), any event-level
  logic (passes, tackles, duels), and the difference between a deflection and a controlled touch beyond the 4 s vote.

## 6. Missing and noisy tracking

Measured on the 158,857 scored frames (`docs/results/missingness_summary.md`):

* The ball is missing on **40.7%** of frames in both halves, in 150 gaps whose median length is about **10 to 12 s**;
  99.5% to 99.9% of the missing frames lie in gaps longer than 5 s. A tracked slot count of 22 goes with 26.6% ball
  missing, 16 or fewer with 88.9%. So the missingness is not random.
* Some minutes (minute 32 of each half) have the ball missing throughout while the video shows the players gathered
  at the touchline: a break, where the ball is out of play.

**Ball recovery was tested and dropped** (`recovery/`, `docs/results/ball_recovery_*.md`). Hidden stretches of tracked
ball (1, 2, 5, 10 s) were reconstructed with last position, constant velocity, linear interpolation, and a Set
Transformer + Bi-LSTM adapted from Kim et al., KDD 2023 (arXiv:2306.08206) in a direct and a residual form, on a
time-ordered split with a clean test block and a protocol written in advance. Linear interpolation was the best method
up to 5 s (mean error 0.8, 2.2 and 6.3 m at 1, 2 and 5 s; about 12 m at 10 s); the neural model was not better at any
length. Because real gaps are mostly far longer than 10 s, recovery would change little, and the submission uses the
hold instead.

## 7. Evidence, and what is not known

* **Reproduction.** Our labels equal the committed reference baselines on all 199,319 frames for the four variants,
  and the official scorer returns the published public-window scores (`docs/results/possession_baseline.md`).
* **Where the reference errs (public window; description, not tuning).** The ball is tracked on 66% of the `D`
  frames, so "ball not tracked means `D`" finds 314 of 1,469 of them; a missing ball is often live play (164 live
  frames wrongly `D`). This motivated the two changes.
* **Stoppage detector.** Evaluated on a grid fixed in advance against the 59 Smart Tagging restarts: it finds 92% of
  corner and 75% of post-goal kick-off stoppages but only 30% to 42% of goal kicks and throw-ins; it calls no frame
  dead in the public window, whose four dead-ball stretches are of the short kind
  (`docs/results/deadball_detector.md`). It is a partial detector.
* **Ablation** (same smoothing, `docs/results/possession_v1b.md`): reference 594 possessions of mean 7.2 s and 32.3%
  `D`; hold only 517 of 12.3 s and 0%; detector only 559 of 7.2 s and 36.9%; hold + detector 526 of 9.6 s
  and 20.8%; **hold + detector + long gaps as `D` (T_E, previous candidate) 493 of 8.0 s and 37.9%**. T_E
  relabels 10.0% of the scored frames relative to the reference. On the public window
  the hold variants score spellF1 0.3610 against 0.3256, a difference smaller than the effect of one possession in a
  window of 17, so it shows nothing.
* **Manual look at 10 long ball gaps (a design input, not labels).** Ten gaps of at least 10 s, evenly spaced by start
  frame (rule fixed before looking; `docs/results/gap_check_list.md`), were watched in the tactical video by a team
  member. All ten were dead ball (goal kick or keeper holding the ball, corner set-up, drinks break, penalty
  stoppage, free kick, goal celebration, yellow card, another stoppage); in none was the ball in play but hidden.
  This led to the `D` rule for 10 s gaps and ended the idea of a video ball detector. It is a sample of 10, and it is a
  human inspection of match video; see section 10. Short gaps (public window) were not inspected and may be live.
* **A segmentation we dropped.** A two-state Viterbi path with a 3 s switch penalty gave 151 possessions of mean 33 s
  over the match and 6 possessions against 17 true ones on the public window (spellF1 0.1913). It was not tuned
  afterwards, and not used.
* **Public window (description; no threshold was chosen from it at this stage).** Variant E: spellF1 0.3707, macroF1 0.6613; reference
  0.3256 and 0.6207; hold variants 0.3610 and 0.5157. With 17 possessions this is within noise.
* **Out-of-play detector (T_E, previous candidate).** A rule fixed in advance from the Laws of the Game
  (`docs/results/out_of_play.md`). On the three sample matches it fired 29 stretches with frame precision 0.59, just
  below the 0.6 we had written as the bar. Two of those matches have almost no tracked dead ball (the ball is blanked
  when out), so they test only false alarms; on the one that resembles ours (game 1) precision was 0.73 and it found
  40% of the tracked dead frames. **We adopted it although it missed our own pre-registered bar; this is a deviation,
  stated here and in the work log.** On our match it adds 39 stretches (median 12 s, 602 s in all): `D` 42.7%,
  493 possessions of 7.4 s. Known flaws: a ball tracked beyond a line during live play (seen in the sample matches)
  starts a false stretch, and a ball that goes out and then stays untracked can run to the 60 s cap.
* **Two candidates for the untracked ball that were built and not submitted.** F, a hand-weighted scorecard per gap
  (`docs/results/gap_scorecard.md`), and G, a logistic regression per gap trained on the three sample matches
  (`docs/results/gap_model.md`). On held-out sample matches G found dead gaps better than our 10 s rule (frame F1 0.874
  against 0.826), but for gaps of 5 s or more (our match's range) it did no better than calling every gap dead, and with
  the out-of-play detector it pushed `D` to 45.8%, beyond our plausibility bound. We kept the 10 s rule.
* **Public window (description only).** T_E: spellF1 0.3905, macroF1 0.7567, dead frames found 48% (all
  correct); E 0.3707 / 0.6613. Only one out-of-play stretch falls in this window, so this shows the rule working once,
  not a general gain.
* **Controlled-ball evidence (current candidate, `possession/submit.py`).** Dead-ball rules are exactly T_E's; only the
  per-frame team evidence changes: a frame votes only when exactly one team has a tracked player within 1.5 m of the
  ball, the vote spans +/-60 frames, and ties keep the previous decided possession. On the full match this gives 395
  possessions (H 196 of mean 11.0 s, A 199 of mean 7.4 s) and the same 42.7% `D`, against T_E's 493 possessions. On the
  public window: spellF1 **0.5212**, macroF1 **0.7715**, 16 predicted possessions against 17 true (T_E: 0.3905 / 0.7567
  / 25). Two diagnostics say where the difference is: keeping the ground truth's live frames and our dead-ball
  frames scores 1.0000, so on this window every `D` we write is correct; keeping our live labels but the ground
  truth's dead-ball frames raises spellF1 to 0.5818, so dead-ball recall is still a source of loss (our `D` recall
  is 0.481, inherited unchanged from T_E). The gain is in live segmentation: ungated nearest-player votes cut long
  possessions at contests and loose balls, and gating removes those cuts. The chosen settings sit on a plateau (gate
  0.8 to 3.0 m and vote window 50 to 70 all give about 0.49 to 0.52 on this window), so the number is not a single lucky
  parameter, but it is still one window of 17 possessions and cannot rank the two files.
* **Evidence layer on other matches (pre-registered, `docs/results/evidence_layer.md`).** On Metrica's open sample
  games 1 and 2, with possession built from their event data and the same dead-ball frames for every method, spellF1
  was: nearest player (T_E) 0.521 / 0.488; one team within 1.5 m 0.562 / 0.599; plus the in-flight gate (7 m/s)
  **0.575 / 0.635** (macroF1 0.933 / 0.950). A nearest-player fallback and a switch-on-reception segmentation rose on
  one game and fell on the other and were dropped, as the rule written in advance required. With the ball blanked
  about 40% of the time the ordering held. On our match the gate gives 334 possessions of mean 10.9 s (public window,
  described after the choice: spellF1 0.5455, macroF1 0.7766, accuracy 0.8127). Limits: two matches, a reference
  built from events rather than frame annotation, and a cleaner tracked ball than ours.
* **Learned evidence models (pre-registered, not adopted; `docs/results/gbm_evidence.md`, `docs/results/rf_shap.md`).**
  On the same sample games, leave one game out, with per-frame features (each team's nearest distance and players
  within 1.5 / 3 / 5 m, their 1 s means, ball speed, time since the ball was seen, and the rule's own label): gradient
  boosting spellF1 0.583 / 0.632 and a random forest 0.572 / 0.641, against the rule's 0.575 / 0.635. Neither is higher
  on both games, so neither replaces the rule. A forest **without** the rule's label is clearly worse (0.542 / 0.555,
  frame accuracy 0.83 against 0.91, and it fragments), and with the ball blanked ~40% it falls to 0.41 / 0.36. SHAP
  on held-out frames: with the rule's label, that label dominates (mean abs SHAP 0.23 / 0.20, next 0.06 / 0.08);
  without it, the top signals are each team's nearest distance to the ball and the players within 1.5 m, the
  quantities the rule uses. Both methods predict about 240 possessions against about 370 true, so the remaining loss
  is segmentation (short possessions merged), not the per-frame evidence.
* **Unknown.** The full-match scores of all candidates. The controlled-ball candidate is chosen because it is the
  challenge's own possession definition and it wins on two independent matches; whether that margin survives
  158,857 graded frames is unknown, and the main new risk is a live phase where tracking loses the controlling player
  for several seconds, so the old team is held.

## 8. Limitations and what we would do with more time

* No validation signal for the final metric; every component is reasoned, none is measured on the full match.
* Hold can extend a possession through a real stoppage inside an untracked stretch shorter than 10 s; and `D` over
  10 s gaps is wrong wherever a long gap is live play (not seen in ten gaps, but only ten were looked at).
* The stoppage detector misses short stoppages where the ball stays on the pitch; the out-of-play detector covers only
  those where the ball is seen beyond a line, and its precision on other matches is about 0.6 to 0.7.
* Possession switches rely on a fixed 1.5 m control gate and a +/-60 frame vote; a switch penalty or a per-frame
  confidence that is justified without ground truth would help. A live phase where the controlling player is briefly
  untracked is held as the old team, which is right for a loose ball but wrong if control changed unseen.
* With more time and permitted labelled data from other matches: a supervised per-frame model for frames where the
  ball is not tracked, with time-ordered validation; video-based ball detection for long gaps.

## 9. Reproduction

`docs/REPRODUCE.md`. In short: `pip install -r requirements-submission.txt`, put the data bundle in `data/`, then
`python possession/submit.py --out <team_name>.csv`. The pipeline is deterministic; for the bundle used in
development the file has SHA-256 `8409a8920f34f0456d68aefdf47b9640804625173bc02d0ec866a2ca870870e5`
(`H 53973  A 37061  D 67823  X 40462`). The previous T_E candidate is still reproducible with
`python possession/make_submission.py --out <team_name>.csv` (SHA-256
`fe38ac194ba507cdf8cc3ed4e2da26e907a2e4b65bf006dc7320f66131e5acb9`). Both use numpy and pandas only;
`python possession/test_pipeline.py` runs ten checks and `python possession/test_submit.py` six more, all without data.

## 10. Rules, data handling and LLM use

* No hand annotation of labels; no event data or any other human-made data about the match; the private ground truth was not
  used. The public 5 minutes were not used for training. They were used as a check, and their score also informed the
  choice between the T_E and controlled-ball candidates (section 7); we disclose that as a deviation from using the
  window only as a check. That choice, and the in-flight gate, were then tested on two other matches with a rule written
  in advance (section 7), and the submitted file is the one that test selected. The data is not in the repository
  (`.gitignore` excludes `data/`), and the repository holds no frame-level export of it. **One human inspection:** a team member watched ten long ball gaps
  in the tactical video and described what happened, which led to the long-gap `D` rule (section 7). No label was
  taken from it. We disclose it so that the organisers can judge it against the rule on hand annotation. On
  2026-11-15 the same member described the same ten gaps again in chat; that description was not used as labels,
  thresholds or validation.
* **LLM use and data handling.** The code and analysis were
  developed with Claude Code, a hosted LLM agent. The scripts ran on our own machine, but their printed output was
  returned to the hosted model as part of the conversation. No data files were uploaded, no labels were produced by the model, and every label comes from the rules in this repository.

## References

* Kim, Y. et al., *Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical
  Bi-LSTM*, KDD 2023, arXiv:2306.08206 (basis of the ball-recovery experiment; we adapted it, see `recovery/README.md`).
* Organisers' repository (`metrica-sports/more-than-a-hack-2027`): data description, scorer, reference baselines.
* Metrica Sports sample data (`metrica-sports/sample-data`): three anonymised matches, used to train candidate G and to
  test the out-of-play rule; not in the submitted labels.
* Our work log, with every finding and its evidence: `docs/worklog.md`; tables with provenance: `docs/results/`.
