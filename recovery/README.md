# recovery/ - filling in the ball where it is not tracked

The ball is missing on 40.7% of the scored frames, in gaps that last seconds (worklog findings 13-16).
This folder tests how well the missing stretches can be reconstructed from the players and from the
ball where it IS tracked, with a model taken from the literature and with simple baselines.

## The task, as an experiment

1. Take a stretch of the match where the ball is tracked.
2. **Hide** a contiguous part of it (1, 2, 5 or 10 seconds), keeping at least 2 s of tracked ball on
   each side, as real gaps have.
3. Ask each method to reconstruct the hidden part from the players' positions and the visible ball.
4. Score against what was hidden. The same trials are used for every method.

No label is made by hand and no human-made data about the match is used. The only "truth" is the
automatically tracked ball itself, which is what makes the approach self-supervised.

## Why the evaluation is built this way (worklog "Design constraints")

| Constraint | How it is met |
|---|---|
| Respect temporal order | Split v2: training = half 1 minutes 0-36 + half 2 minutes 0-20; validation = half 2 minutes 21-31; `test_seen` = half 2 minutes 32-end; clean `test` = half 1 minutes 38-end; 1-minute buffers between periods are unused. Windows never cross a half or a split boundary. Frames 1/25 s apart are near duplicates, so random splits would leak. |
| Gaps like real gaps | Contiguous hidden stretches of seconds, not single frames; during training 1-25 s, plus whole-ball-hidden windows. |
| Use only what a real gap offers | Tracked ball on both sides of the hidden stretch (two-sided model); a past-only variant (`--causal`) is the stricter comparison. |
| Do not pick the model on the test set | Validation chooses the epoch; the clean test period is scored once at the end. The first model's test period was seen, so v2 keeps it as `test_seen` (flagged) and takes a new block out of training. Protocol and decision rule were written in `docs/worklog.md` before the runs. |
| Teams swap ends at half-time | Training windows are mirrored at random (horizontal and vertical flips), as in the paper. |
| Provenance | Every results file records the command, git commit, SHA-256 of every code file and of the input tracking, library versions, device and seeds. |

## Methods compared

| Method | Uses | File |
|---|---|---|
| last position | the last tracked ball position, held | `baselines.py` |
| constant velocity | the last two tracked positions, continued and clipped to the pitch | `baselines.py` |
| linear interpolation | a straight line between the tracked ball before and after the gap | `baselines.py` |
| Set Transformer + hierarchical Bi-LSTM, direct | the players as a set, plus the ball where visible; outputs the ball | `model.py` |
| same, residual on interpolation (`--residual`) | as above, but outputs a correction added to linear interpolation and receives the interpolation as input | `model.py` |

### Findings so far

On the clean test the residual model is not better than linear interpolation at any gap length
(1, 2, 5, 10 s), and the direct model is worse up to 5 s; the best method for hidden stretches of up
to 10 s is plain linear interpolation, and nothing tried recovers a 10 s gap well (the best methods
are at 11-14 m, the past-only baselines at 22-24 m).
Details, intervals and caveats: `docs/worklog.md` ("2026-11-12 (result)") and `docs/results/`.

### The model and the paper

The model follows Kim et al., "Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set
Transformer and Hierarchical Bi-LSTM", KDD 2023
([arXiv:2306.08206](https://arxiv.org/abs/2306.08206)). Our notes on the paper come from a read of
its arXiv HTML version; part of its appendices was only available as a model-extracted summary, so
anything that matters (the loss definitions in particular) should be checked against the paper.

| | Paper | Here | Why |
|---|---|---|---|
| Players | 22 ordered players, Set Transformer per team and overall | up to 46 track slots as a set, with a padding mask and an always-present "nobody has it" token | our slots are not stable identities and are often missing |
| Features | location, velocity, speed, acceleration | the same, plus a presence flag and a team embedding | the paper's ablation found each feature helps |
| Hierarchy | stage 1 predicts who possesses the ball, stage 2 predicts where it is, both Bi-LSTMs | the same two stages; possession is an attention over the players at each step | |
| Possession targets | built from human-annotated event data | **algorithmic pseudo-labels**: the nearest tracked player within 3 m of the tracked ball, else nobody | `RULES.md` forbids human labels and event data for this match |
| Losses | MSE + reality loss + cross-entropy (weight 20) | smooth-L1 on hidden stretches + cross-entropy (weight 1) | the reality-loss formula as printed has an unresolved sign; not copied |
| Window / rate | 10 s at 10 Hz | 30 s at 5 Hz | our gaps are seconds to minutes long |
| Size | 8.7M parameters, LSTM 2 x 256 | about 1M parameters, LSTM 2 x 128 | one match of data, not a dozen |
| Masking for imputation | random 80% masking; gap lengths not stated | contiguous stretches of 1-25 s, or all of the ball (25% of windows) | worklog finding 16 |

## Run

Needs PyTorch, which is not in `requirements.txt`. It lives in a local virtual environment:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install torch
.venv/bin/python recovery/run.py --epochs 50 --patience 10             # direct output
.venv/bin/python recovery/run.py --residual --epochs 50 --patience 10  # correction on interpolation
.venv/bin/python recovery/run.py --causal                              # past-only variant (not yet run)
.venv/bin/python recovery/run.py --quick                               # a few seconds, to check the pipeline
```

Outputs: `docs/results/ball_recovery_<bilstm|causal>.md` (tables, set-up and provenance; committed)
and `recovery/output/` (per-trial CSV, training log, weights; git-ignored).

## Files

| File | Purpose |
|---|---|
| `data.py` | tracking to 5 Hz arrays, time-ordered split, training windows and masks, fixed evaluation trials |
| `baselines.py` | the three model-free methods |
| `model.py` | the Set Transformer and the two-stage Bi-LSTM, plus the possession pseudo-labels |
| `run.py` | training, evaluation, bootstrap confidence intervals, results file with provenance |
| `output/` | local, git-ignored |

## Limits to keep in mind

* One match; the test period is about 21 minutes of the second half, so a few dozen trials per gap
  length. The tables give bootstrap confidence intervals and paired differences, and differences
  inside the interval should not be read as real.
* Team agreement is a proxy computed from the tracking (nearest tracked player to the ball), not a
  possession ground truth.
* The evaluation hides stretches where the ball WAS tracked. Real gaps are not random (worklog
  finding 15: they come with fewer tracked players, near the goals and when the ball is slow), and
  some are stoppages with no ball to find at all (finding 17). Performance on real gaps may be worse.
* Rules: no hand labels, no event data, no use of the public ground truth. Model weights and
  per-trial tables are derived from gated data and stay out of git.
