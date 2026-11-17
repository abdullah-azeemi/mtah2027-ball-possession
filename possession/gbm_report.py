"""Gradient-boosted per-frame evidence (GB) against the submitted rule (CS), leave one sample game out.

Pre-registered in docs/worklog.md ("2026-11-16 (gradient-boosted evidence, GB)"). Writes docs/results/gbm_evidence.md.

    python possession/gbm_report.py          # needs scikit-learn (not part of the submission environment)

Metrica open sample games 1 and 2 only; reference possession and oracle dead ball as in evidence_report.py.
Aggregates only.
"""

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import submit  # noqa: E402
import evidence  # noqa: E402
import evidence_report as er  # noqa: E402
import gapmodel  # noqa: E402
import possession_metrics as pm  # noqa: E402

RESULT_MD = os.path.join(ROOT, "docs", "results", "gbm_evidence.md")
SMOOTH = 60
ROLL = 12
FEATURES = ("near_h", "near_a", "h15", "a15", "h3", "a3", "h5", "a5", "ball_speed", "ball_tracked",
            "since_tracked", "near_h_roll", "near_a_roll", "h3_roll", "a3_roll", "cs_label")


def rolling_mean(x, k):
    """Centred mean over +/-k frames, ignoring NaN."""
    ok = ~np.isnan(x)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    c = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(x))
    lo, hi = np.maximum(0, idx - k), np.minimum(len(x), idx + k + 1)
    n = c[hi] - c[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (s[hi] - s[lo]) / np.maximum(n, 1), np.nan)


def features(pos, ball, team, halves):
    n = len(ball)
    d = np.sqrt(((pos - ball[:, None, :]) ** 2).sum(axis=2))      # (n, P), NaN where either is missing
    cols = {}
    for name, code in (("h", 0), ("a", 1)):
        dt = np.where(np.isnan(d[:, team == code]), np.inf, d[:, team == code])
        near = dt.min(axis=1)
        tracked = ~np.isnan(ball[:, 0])
        cols["near_" + name] = np.where(tracked, np.minimum(near, 30.0), np.nan)
        for r, key in ((1.5, "15"), (3.0, "3"), (5.0, "5")):
            cols[name + key] = np.where(tracked, (dt <= r).sum(axis=1), np.nan).astype(float)
    tracked = ~np.isnan(ball[:, 0])
    cols["ball_speed"] = np.minimum(submit.ball_speed(ball), 40.0)
    cols["ball_tracked"] = tracked.astype(float)
    last = np.maximum.accumulate(np.where(tracked, np.arange(n), -10 ** 9))
    cols["since_tracked"] = np.minimum((np.arange(n) - last) / evidence.FPS, 30.0)
    for key in ("near_h", "near_a", "h3", "a3"):
        cols[key + "_roll"] = rolling_mean(cols[key], ROLL)
    cs = evidence.team_labels("CS", pos, ball, team, halves)
    cols["cs_label"] = np.where(cs >= 0, cs, np.nan).astype(float)
    return np.column_stack([cols[f] for f in FEATURES])


def gb_labels(model, X, halves, n):
    p = model.predict_proba(X)[:, list(model.classes_).index(0)]       # p(home)
    out = np.full(n, -1, dtype=np.int8)
    for first, last in halves:
        ps = rolling_mean(p[first:last + 1], SMOOTH)
        out[first:last + 1] = np.where(ps > 0.5, 0, 1)
    return out


def to_pred(tl, ref):
    pred = np.where(tl == 1, pm.A, pm.H).astype(np.int8)
    pred[ref == pm.D] = pm.D
    pred[ref == pm.X] = pm.X
    return pred


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    games = {}
    for g in (1, 2):
        pos, team, ball, dead, halves = gapmodel.load_sample_csv(g)
        ref = er.reference_labels(g, len(ball), dead, halves)
        games[g] = dict(pos=pos, team=team, ball=ball, halves=halves, ref=ref,
                        X=features(pos, ball, team, halves),
                        Xd=features(pos, er.dropout(ball), team, halves), ball_d=er.dropout(ball))
        print("game", g, "features", games[g]["X"].shape)

    results, importance = {}, {}
    for test in (1, 2):
        train = 3 - test
        tr = games[train]
        live = (tr["ref"] == pm.H) | (tr["ref"] == pm.A)
        y = np.where(tr["ref"][live] == pm.H, 0, 1)
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
        model.fit(tr["X"][live], y)
        te = games[test]
        for cond, X, b in (("tracked", te["X"], te["ball"]), ("dropout", te["Xd"], te["ball_d"])):
            n = len(b)
            gb = to_pred(gb_labels(model, X, te["halves"], n), te["ref"])
            cs = to_pred(evidence.team_labels("CS", te["pos"], b, te["team"], te["halves"]), te["ref"])
            for name, pred in (("CS", cs), ("GB", gb)):
                results[(test, cond, name)] = er.score(te["ref"], pred)
                print(test, cond, name, {k: round(v, 4) if isinstance(v, float) else v
                                         for k, v in results[(test, cond, name)].items()})
        # frame-level held-out accuracy of the raw classifier on live frames
        lv = (te["ref"] == pm.H) | (te["ref"] == pm.A)
        raw = model.predict(te["X"][lv])
        importance[test] = float((raw == np.where(te["ref"][lv] == pm.H, 0, 1)).mean())

    lines = ["# Gradient-boosted evidence (GB) against the submitted rule (CS)", "",
             "Generated by `possession/gbm_report.py`. Pre-registered in `docs/worklog.md` "
             "(\"2026-11-16 (gradient-boosted evidence, GB)\") before any model was fitted. Leave one Metrica sample "
             "game out; event-derived reference and oracle dead ball as in `docs/results/evidence_layer.md`. "
             "No data from our match is used.", "", "## Provenance", "",
             "- generated (UTC): {}".format(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
             "- git commit: `{}`".format(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                        text=True, cwd=ROOT).stdout.strip()),
             "- possession/gbm_report.py sha256: `{}`".format(sha(os.path.abspath(__file__))),
             "- possession/evidence.py sha256: `{}`".format(sha(os.path.join(HERE, "evidence.py"))),
             "- model: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0); "
             "smoothing: mean p over +/-{} frames".format(SMOOTH),
             "- features: {}".format(", ".join(FEATURES)),
             "- python {} / numpy {} / scikit-learn {}".format(sys.version.split()[0], np.__version__,
                                                               __import__("sklearn").__version__), ""]
    for cond, title in (("tracked", "Held-out game, ball as tracked (primary)"),
                        ("dropout", "Held-out game, ball blanked ~40% (robustness)")):
        lines += ["## {}".format(title), "",
                  "| test game | method | spellF1 | 95% band | macroF1 | transF1@25 | team acc | possessions (pred / true) |",
                  "|---|---|---|---|---|---|---|---|"]
        for g in (1, 2):
            for m in ("CS", "GB"):
                r = results[(g, cond, m)]
                lines.append("| {} | {} | {:.4f} | {:.3f}-{:.3f} | {:.4f} | {:.4f} | {:.4f} | {} / {} |".format(
                    g, m, r["spell"], r["lo"], r["hi"], r["macro"], r["trans"], r["team"], r["n_pred"], r["n_gt"]))
        lines.append("")
    lines += ["Raw classifier, held-out live frames, before smoothing: accuracy {:.4f} (test game 1), {:.4f} "
              "(test game 2).".format(importance[1], importance[2]), ""]
    up = all(results[(g, "tracked", "GB")]["spell"] > results[(g, "tracked", "CS")]["spell"] for g in (1, 2))
    mac = all(results[(g, "tracked", "GB")]["macro"] >= results[(g, "tracked", "CS")]["macro"] - 0.01 for g in (1, 2))
    flag = all(results[(g, "dropout", "GB")]["spell"] < results[(g, "dropout", "CS")]["spell"] - 0.02 for g in (1, 2))
    lines += ["## Decision rule (written in advance)", "",
              "- spellF1 higher than CS on both held-out games: **{}**".format("yes" if up else "no"),
              "- macroF1 not lower than CS by more than 0.01 on either: **{}**".format("yes" if mac else "no"),
              "- dropout flag (loses > 0.02 spellF1 on both): **{}**".format("FLAG" if flag else "no"),
              "- **GB replaces CS: {}**".format("yes" if up and mac else "no"), ""]
    with open(RESULT_MD, "w") as handle:
        handle.write("\n".join(lines))
    print("wrote", RESULT_MD)


if __name__ == "__main__":
    main()
