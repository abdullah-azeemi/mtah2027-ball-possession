"""Train and evaluate ball recovery, and write the results with their provenance.

    .venv/bin/python recovery/run.py --residual --epochs 50 --patience 10   # correction on top of interpolation
    .venv/bin/python recovery/run.py --epochs 50 --patience 10              # direct output (the first model)
    .venv/bin/python recovery/run.py --causal                               # past-only variant (stricter)
    .venv/bin/python recovery/run.py --quick                                # tiny run to check the pipeline

What is compared, on identical fixed trials (recovery/data.py: a hidden stretch of 1, 2, 5 or 10 s
inside tracked-ball context, in time-ordered periods):

    last position / constant velocity / linear interpolation   (no learning, from baselines.py)
    the Set Transformer + hierarchical Bi-LSTM model            (model.py), direct or residual

The model is trained only on the TRAINING period. The validation period chooses the epoch and is the
only period any decision may look at. Two test periods are scored once, at the end, with that epoch:

    test       clean: never used for any decision (split v2, first half minutes 38-end)
    test_seen  looked at once for the first model version: reported but NOT clean evidence

Nothing here uses a ground-truth possession label or any human-made data about the match: the
training target is the automatically tracked ball.

Outputs
    docs/results/ball_recovery_<tag>.md       tables + configuration + provenance   (committed)
    recovery/output/                          per-trial CSV, training log, weights  (git-ignored)
"""

import argparse
import datetime
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import baselines  # noqa: E402
import data  # noqa: E402
import model as model_lib  # noqa: E402
import missingness_summary as ms  # noqa: E402  (provenance helpers: sha256, git, pipe_table)
from games import asset_paths  # noqa: E402

OUT_DIR = os.path.join(HERE, "output")
RESULTS_DIR = os.path.join(ROOT, "docs", "results")
TRIAL_SEED = 12345      # fixes the evaluation trials, independent of the training seed
GAP_SECONDS = (1, 2, 5, 10)
# Order matters: trials are drawn from one seeded stream in this order, so "val" and "test_seen"
# get exactly the trials they had in split v1 (docs/results/ball_recovery_bilstm.md).
SPLITS = ("val", "test_seen", "test")
REFERENCE = "linear interpolation (two-sided)"
SPLIT_LABELS = {
    "test": "TEST, clean (first half minutes 38-end; no decision has ever looked at it; scored once)",
    "test_seen": "Earlier test period (second half minutes 32-end): looked at once for the first model "
                 "version, so NOT clean evidence",
    "val": "Validation (the only period used to choose the epoch)",
}


# Scoring -----------------------------------------------------------------------------------------
def team_of_nearest(pos_steps, points, team):
    """Team (0/1) of the tracked player nearest to `points` at each step; -1 when nobody is tracked."""
    d = np.linalg.norm(pos_steps - points[:, None, :], axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    return np.where(np.isfinite(d.min(axis=1)), team[d.argmin(axis=1)], -1)


def score_trial(pred, truth, pos_gap, team):
    """(mean position error in m, share of steps where the nearest player's team matches truth's)."""
    pe = float(np.linalg.norm(pred - truth, axis=1).mean())
    t_pred, t_true = team_of_nearest(pos_gap, pred, team), team_of_nearest(pos_gap, truth, team)
    valid = (t_pred >= 0) & (t_true >= 0)
    return pe, float((t_pred[valid] == t_true[valid]).mean()) if valid.any() else np.nan


def model_predictions(net, pos, shown, team, device, batch=64):
    """Model output for every window, (N, WINDOW, 2) metres."""
    net.eval()
    team_t = torch.as_tensor(team, device=device)
    out = []
    with torch.no_grad():
        for i in range(0, len(pos), batch):
            p = torch.as_tensor(pos[i:i + batch], device=device)
            b = torch.as_tensor(shown[i:i + batch], device=device)
            hidden = ~torch.isfinite(b[..., 0])
            out.append(net(p, b, hidden, team_t)[0].float().cpu().numpy())
    return np.concatenate(out)


def evaluate(trials, pos, truth, team, net, device):
    """Per-trial rows for every method on the same trials."""
    shown = data.trial_input_ball(truth, trials)
    predicted = {name: [fn(shown[i], t["gap_start"], t["gap_end"]) for i, t in enumerate(trials)]
                 for name, fn in baselines.BASELINES.items()}
    if net is not None:
        full = model_predictions(net, pos, shown, team, device)
        predicted[net.label] = [full[i, t["gap_start"]:t["gap_end"]] for i, t in enumerate(trials)]
    rows = []
    for name, guesses in predicted.items():
        for i, (t, guess) in enumerate(zip(trials, guesses)):
            sl = slice(t["gap_start"], t["gap_end"])
            pe, tpa = score_trial(guess, truth[i, sl], pos[i, sl], team)
            rows.append({"trial": i, "method": name, "seconds": t["seconds"], "pe_m": pe, "team_agree": tpa})
    return pd.DataFrame(rows)


def mean_pe_over_lengths(frame, method):
    sub = frame[frame["method"] == method]
    return float(sub.groupby("seconds")["pe_m"].mean().mean())


def bootstrap_ci(values, rng, n=2000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


# Training ----------------------------------------------------------------------------------------
def train(args, arrays, segments, val, team, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = model_lib.BallRecoverer(bidirectional=not args.causal, residual=args.residual).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    team_t = torch.as_tensor(team, device=device)
    method = net.label

    log, best, best_state, since_best = [], np.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        started, losses = time.time(), []
        for _ in range(args.steps):
            pos, ball = data.sample_windows(arrays, segments["train"], rng, args.batch)
            hidden = data.training_mask(ball, rng)
            ball_in = np.where(hidden[..., None], np.nan, ball)
            pos_t, ball_t = torch.as_tensor(pos, device=device), torch.as_tensor(ball, device=device)
            ball_in_t = torch.as_tensor(ball_in, device=device)
            hidden_t = torch.as_tensor(hidden, device=device)

            pred, logits = net(pos_t, ball_in_t, hidden_t, team_t)
            target = hidden_t & torch.isfinite(ball_t[..., 0])      # tracked in truth, hidden from input
            recon = (F.smooth_l1_loss(pred[target] / 10.0, ball_t[target] / 10.0)
                     if target.any() else pred.sum() * 0.0)
            labels = model_lib.possession_pseudo_labels(pos_t, ball_t)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            loss = recon + args.lambda_ce * ce
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            losses.append((float(recon.detach()), float(ce.detach())))

        frame = evaluate(val["trials"], val["pos"], val["truth"], team, net, device)
        val_pe = mean_pe_over_lengths(frame, method)
        recon_mean, ce_mean = np.mean(losses, axis=0)
        log.append({"epoch": epoch, "train_recon": recon_mean, "train_ce": ce_mean, "val_pe_m": val_pe,
                    "seconds": time.time() - started})
        print("epoch {:>2}  train recon {:.3f}  ce {:.3f}  | val PE {:.2f} m  ({:.0f}s)".format(
            epoch, recon_mean, ce_mean, val_pe, time.time() - started), flush=True)
        if val_pe < best:
            best, since_best = val_pe, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            since_best += 1
            if since_best >= args.patience:
                print("early stop: no validation improvement for {} epochs".format(args.patience))
                break
    net.load_state_dict(best_state)
    return net, pd.DataFrame(log), n_params, method


# Reporting ---------------------------------------------------------------------------------------
def summary_tables(frame, methods, reference, rng):
    """Mean PE with a bootstrap CI per method and gap length, team agreement, and paired deltas."""
    rows, tpa_rows, delta_rows = [], [], []
    for seconds in sorted(frame["seconds"].unique()):
        sub = frame[frame["seconds"] == seconds]
        row = {"gap": "{} s".format(seconds), "trials": int((sub["method"] == methods[0]).sum())}
        tpa_row = {"gap": "{} s".format(seconds)}
        for m in methods:
            v = sub[sub["method"] == m]["pe_m"].values
            lo, hi = bootstrap_ci(v, rng)
            row[m] = "{:.2f} [{:.2f}, {:.2f}]".format(v.mean(), lo, hi)
            tpa_row[m] = "{:.1f}%".format(100 * np.nanmean(sub[sub["method"] == m]["team_agree"].values))
        rows.append(row)
        tpa_rows.append(tpa_row)
        ref = sub[sub["method"] == reference].set_index("trial")["pe_m"]
        delta = {"gap": "{} s".format(seconds)}
        for m in methods:
            if m == reference:
                continue
            d = (sub[sub["method"] == m].set_index("trial")["pe_m"] - ref).values
            lo, hi = bootstrap_ci(d, rng)
            delta[m] = "{:+.2f} [{:+.2f}, {:+.2f}]".format(d.mean(), lo, hi)
        delta_rows.append(delta)
    return pd.DataFrame(rows), pd.DataFrame(tpa_rows), pd.DataFrame(delta_rows)


def provenance(args, device, n_params, runtime):
    paths = asset_paths()
    code = ["data.py", "baselines.py", "model.py", "run.py"]
    lines = [
        "- generated (UTC): {}".format(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python {}`".format(" ".join(sys.argv)),
        "- git commit: `{}`{}".format(ms.git("rev-parse", "HEAD"),
                                      " (tracked files had uncommitted changes)"
                                      if ms.git("status", "--porcelain", "--untracked-files=no") else ""),
    ]
    for name in code:
        lines.append("- recovery/{} sha256: `{}`".format(name, ms.sha256(os.path.join(HERE, name))))
    lines += [
        "- input tracking sha256: `{}`".format(ms.sha256(paths["atd_raw"])),
        "- input tracking metadata sha256: `{}`".format(ms.sha256(paths["atd_meta"])),
        "- python {} / numpy {} / pandas {} / torch {} / device {}".format(
            platform.python_version(), np.__version__, pd.__version__, torch.__version__, device),
        "- split version {}; seeds: training {}, evaluation trials {}".format(
            data.SPLIT_VERSION, args.seed, TRIAL_SEED),
        "- model parameters: {:,}; training + evaluation time {:.0f} s".format(n_params, runtime),
    ]
    return "\n".join(lines)


def describe_split(split_info):
    """The time-ordered split as one sentence per period, taken from data.SPLIT_MINUTES."""
    out = []
    for name, parts in data.SPLIT_MINUTES.items():
        spans = " + ".join("half {} minutes {}-{}".format(h, a, "end" if b is None else b)
                           for h, a, b in parts)
        out.append("{} = {} ({:.1f} min)".format(name, spans, split_info[name] / data.MINUTE))
    return "; ".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps", type=int, default=40, help="batches per epoch")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-ce", type=float, default=1.0, help="weight of the possession loss")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--causal", action="store_true", help="past-only LSTM instead of Bi-LSTM")
    parser.add_argument("--residual", action="store_true",
                        help="output a correction on top of linear interpolation instead of the ball directly")
    parser.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.quick:
        args.epochs, args.steps, args.batch = 2, 4, 4
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    tag = "{}_{}_split2{}".format("causal" if args.causal else "bilstm",
                                  "residual" if args.residual else "direct",
                                  "_quick" if args.quick else "")
    started = time.time()

    print("loading tracking at 5 Hz ...")
    arrays, team = data.load_halves()
    segments = data.split_segments(arrays)
    split_info = {k: sum(e - s for _, s, e in v) for k, v in segments.items()}
    print("steps per split:", split_info, "(5 steps = 1 s)")

    rng_trials = np.random.default_rng(TRIAL_SEED)
    split_data = {}
    for split in SPLITS:
        trials = data.make_trials(arrays, segments[split], rng_trials, GAP_SECONDS)
        pos, truth = data.trial_arrays(arrays, trials)
        split_data[split] = {"trials": trials, "pos": pos, "truth": truth}
        counts = pd.Series([t["seconds"] for t in trials]).value_counts().sort_index().to_dict()
        print("{} trials by gap seconds: {}".format(split, counts))

    net, log, n_params, method = train(args, arrays, segments, split_data["val"], team, device)

    os.makedirs(OUT_DIR, exist_ok=True)
    log.to_csv(os.path.join(OUT_DIR, "training_log_{}.csv".format(tag)), index=False)
    torch.save(net.state_dict(), os.path.join(OUT_DIR, "model_{}.pt".format(tag)))

    results = {}
    for split in SPLITS:
        d = split_data[split]
        results[split] = evaluate(d["trials"], d["pos"], d["truth"], team, net, device)
        results[split].to_csv(os.path.join(OUT_DIR, "trials_{}_{}.csv".format(split, tag)), index=False)

    methods = list(baselines.BASELINES) + [method]
    boot = np.random.default_rng(7)
    sections = []
    for split in ("test", "test_seen", "val"):
        pe, tpa, delta = summary_tables(results[split], methods, REFERENCE, boot)
        sections.append((split, pe, tpa, delta))

    runtime = time.time() - started
    prov = provenance(args, device, n_params, runtime)
    write_markdown(args, tag, sections, log, split_info, split_data, prov, method)
    print("\nCLEAN TEST period, mean position error (m) [95% bootstrap CI] over the hidden stretch:")
    print(sections[0][1].to_string(index=False))


def write_markdown(args, tag, sections, log, split_info, split_data, prov, method):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "ball_recovery_{}.md".format(tag))
    best = log.loc[log["val_pe_m"].idxmin()]
    parts = ["""# Ball recovery: filling hidden ball stretches ({tag})

Generated by `recovery/run.py`. Method and rationale: `recovery/README.md`. Everything is computed
from the tracking only; no ground-truth possession label, event data or hand labelling is used.

## Provenance

{prov}

## Set-up

* Task: hide a stretch of ball that IS tracked (1, 2, 5 or 10 s), with at least 2 s of tracked ball on
  each side, and reconstruct it from the players and the surrounding ball. Score against what was
  hidden. Same trials for every method.
* Time-ordered split {version} (minutes counted from the first graded frame of a half; 1-minute buffers
  between periods are unused): {split}. Teams swap ends at half-time, so training windows are
  mirrored at random (horizontal and vertical flips).
* Only `val` was used to make any decision (the epoch). `test` is clean. `test_seen` is the test
  period of split v1: it was scored once for the first model version, so results on it are reported
  but are not untouched evidence.
* Trials: validation {nv}, test_seen {ns}, test {nt}.
* Model: {method}, {epochs} epochs run (best epoch {best_epoch} by validation error, {best_val:.2f} m);
  trained on 30 s windows at 5 Hz with randomly hidden stretches (1-25 s, or the whole ball hidden in
  25% of windows). The loss is on hidden stretches where the truth is tracked, plus an auxiliary
  possession loss on algorithmic pseudo-labels (nearest player within 3 m of the tracked ball).
  Learning rate {lr}, batch {batch}, {steps} batches per epoch, possession weight {lam}.
* Metrics: position error = mean distance in metres over the hidden steps. Team agreement = share of
  hidden steps where the team of the tracked player nearest to the guess equals that of the player
  nearest to the true ball. This is a proxy built from the tracking, NOT a possession ground truth.
""".format(tag=tag, prov=prov, version=data.SPLIT_VERSION, split=describe_split(split_info),
           nv=len(split_data["val"]["trials"]), ns=len(split_data["test_seen"]["trials"]),
           nt=len(split_data["test"]["trials"]), method=method, epochs=len(log),
           best_epoch=int(best["epoch"]), best_val=best["val_pe_m"], lr=args.lr, batch=args.batch,
           steps=args.steps, lam=args.lambda_ce)]
    for split, pe, tpa, delta in sections:
        parts.append("## {}\n\n### Position error, metres, mean [95% bootstrap CI over trials]\n\n{}\n\n"
                     "### Team agreement\n\n{}\n\n### Paired difference vs. {} (negative = better)\n\n{}\n"
                     .format(SPLIT_LABELS[split], ms.pipe_table(pe), ms.pipe_table(tpa), REFERENCE,
                             ms.pipe_table(delta)))
    with open(path, "w") as handle:
        handle.write("\n".join(parts))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
