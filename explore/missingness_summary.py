"""How much of the tracking is missing, and over what range, on the frames that are scored.

Question this answers (worklog 2026-11-11, "next"): on the playable (graded) frames, how much of the
ball track and of the player tracks is missing, in what range does that vary from minute to minute
and between the two halves, and how long do the gaps last? The answer decides whether short-gap
interpolation can work at all and how a recovery method has to be validated.

    python explore/missingness_summary.py

Writes
    docs/results/missingness_summary.md       tables + a provenance block (committed)
    explore/output/missingness_per_minute.csv per-minute statistics          (git-ignored)
    explore/output/missingness_overview.png   the same as a picture          (git-ignored)

Only the tracking text and its metadata are read. The ground truth is not touched, so nothing here
is tuned on labels.

Definitions
-----------
* graded frames  = the two halves, video frames 11185-90651 and 110808-190197 (games.py). All
                   statistics below are over graded frames only; the rest of the file is not
                   playing time.
* "playable"     = graded. Graded frames include dead-ball stretches, because only the public
                   5 minutes are labelled; we cannot separate dead ball from live ball elsewhere.
* second half    = graded from 110808, but kick-off is at 113742; the 2,934 frames in between
                   are all dead ball (FAQ.md). They are reported separately.
* tracked slot   = a track slot with an x,y position in that frame (46 slots exist).
* ball missing   = the ball column is NaN.
* minute         = 1,500 consecutive graded frames counted from the start of the half.
* gap            = a maximal run of consecutive frames with the ball missing, inside one half.

Provenance is written into the output: command, UTC time, git commit and whether the working tree
was dirty, SHA-256 of this script and of the input files, library versions.
"""

import argparse
import datetime
import hashlib
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
import frame_explorer as fe  # noqa: E402  (only for the slot -> team mapping)

FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
N_SLOTS = 46
MINUTE = 60 * FPS
# Second-half kick-off (DATA.md / FAQ.md): frames 110808..113741 are graded but all dead ball.
KICKOFF_2ND_VIDEO_FRAME = 113742

RESULT_MD = os.path.join(ROOT, "docs", "results", "missingness_summary.md")
OUT_DIR = os.path.join(HERE, "output")


# Loading -----------------------------------------------------------------------------------------
def load_presence(atd_path):
    """(ball_visible[N], present[N, 46]) from the raw tracking text.

    present[i, k] is True when slot k + 1 has an x,y position at video frame i.
    """
    ball_visible = np.zeros(N_FRAMES, dtype=bool)
    present = np.zeros((N_FRAMES, N_SLOTS), dtype=bool)
    with open(atd_path, "r") as handle:
        for i, line in enumerate(handle):
            frame_id, slots, ball = line.rstrip("\n").split(":")
            if int(frame_id) != i + 1:
                raise ValueError("line {} holds frame_id {}".format(i, frame_id))
            ball_visible[i] = not ball.startswith("NaN")
            present[i] = [not token.startswith("NaN") for token in slots.split(";")]
    return ball_visible, present


def runs(mask):
    """Lengths of the maximal runs of True in a boolean array."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    change = np.diff(padded)
    return np.flatnonzero(change == -1) - np.flatnonzero(change == 1)


# Analysis ----------------------------------------------------------------------------------------
def pct(x):
    return "{:.1f}%".format(100.0 * x)


def pipe_table(frame):
    """A DataFrame as a Markdown pipe table (no extra dependency)."""
    cols = list(frame.columns)
    rows = ["| " + " | ".join(str(c) for c in cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(rows)


def analyse(ball_visible, present, side_of_slot):
    """All the tables, as a dict of DataFrames, plus the per-minute frame for plotting."""
    blocks = half_video_blocks(GAME)  # [(half, first, last)]
    tracked = present.sum(axis=1)
    home = present[:, side_of_slot == "H"].sum(axis=1)
    away = present[:, side_of_slot == "A"].sum(axis=1)

    # Row selectors over graded frames.
    def select(half=None, from_frame=None):
        mask = np.zeros(N_FRAMES, dtype=bool)
        for h, first, last in blocks:
            if half in (None, h):
                mask[max(first, from_frame or first):last + 1] = True
        return mask

    groups = [("Half 1", select(1)),
              ("Half 2 (all graded)", select(2)),
              ("Half 2 from kick-off", select(2, KICKOFF_2ND_VIDEO_FRAME)),
              ("Both halves", select())]

    # A. headline rates per half.
    rows = []
    for name, mask in groups:
        rows.append({"frames": name, "n": int(mask.sum()),
                     "ball missing": pct(1 - ball_visible[mask].mean()),
                     "mean slots tracked": "{:.1f}".format(tracked[mask].mean()),
                     "mean Barcelona": "{:.1f}".format(home[mask].mean()),
                     "mean Athletic": "{:.1f}".format(away[mask].mean()),
                     "min..max slots": "{}..{}".format(tracked[mask].min(), tracked[mask].max())})
    headline = pd.DataFrame(rows)

    # B. distribution of tracked-slot count.
    bins = [("<=16", lambda t: t <= 16), ("17", lambda t: t == 17), ("18", lambda t: t == 18),
            ("19", lambda t: t == 19), ("20", lambda t: t == 20), ("21", lambda t: t == 21),
            ("22", lambda t: t == 22), (">=23", lambda t: t >= 23)]
    rows = []
    for label, test in bins:
        row = {"slots tracked": label}
        for name, mask in groups[:2] + groups[3:]:
            row[name] = pct(test(tracked[mask]).mean())
        # Ball missing rate inside that bin (both halves), the link between the two missingness.
        both = groups[3][1] & test(tracked)
        row["ball missing in this bin (both halves)"] = pct(1 - ball_visible[both].mean()) if both.any() else "-"
        rows.append(row)
    slot_distribution = pd.DataFrame(rows)

    # C. per-minute statistics.
    records = []
    for half, first, last in blocks:
        index = np.arange(first, last + 1)
        minute = (index - first) // MINUTE
        frame = pd.DataFrame({"half": half, "minute": minute,
                              "ball_missing": 1.0 - ball_visible[index],
                              "tracked": tracked[index]})
        records.append(frame.groupby(["half", "minute"]).agg(
            frames=("tracked", "size"), ball_missing=("ball_missing", "mean"),
            tracked=("tracked", "mean")).reset_index())
    per_minute = pd.concat(records, ignore_index=True)

    rows = []
    for name, half in (("Half 1", 1), ("Half 2", 2), ("Both halves", None)):
        part = per_minute if half is None else per_minute[per_minute["half"] == half]
        full = part[part["frames"] == MINUTE]  # drop the partial last minute from the quantiles
        q = full["ball_missing"].quantile([0, .25, .5, .75, 1.0]).values
        rows.append({"minutes (full)": "{} ({})".format(len(part), len(full)),
                     "scope": name,
                     "ball missing per minute: min": pct(q[0]), "25%": pct(q[1]), "median": pct(q[2]),
                     "75%": pct(q[3]), "max": pct(q[4]),
                     "minutes <10% missing": int((full["ball_missing"] < 0.10).sum()),
                     "minutes >75% missing": int((full["ball_missing"] > 0.75).sum()),
                     "mean slots/min: min..max": "{:.1f}..{:.1f}".format(
                         full["tracked"].min(), full["tracked"].max())})
    minute_stats = pd.DataFrame(rows)

    # D. ball gaps and tracked runs, per half (never across half-time).
    rows = []
    for half, first, last in blocks:
        missing = ~ball_visible[first:last + 1]
        gaps, seen = runs(missing), runs(~missing)
        total = gaps.sum()
        rows.append({
            "half": half,
            "gaps": len(gaps),
            "gap length, median (s)": "{:.1f}".format(np.median(gaps) / FPS),
            "p90 (s)": "{:.1f}".format(np.percentile(gaps, 90) / FPS),
            "max (s)": "{:.1f}".format(gaps.max() / FPS),
            "gaps <=1 s": int((gaps <= FPS).sum()),
            "share of missing frames in gaps >5 s": pct(gaps[gaps > 5 * FPS].sum() / total),
            "in gaps >30 s": pct(gaps[gaps > 30 * FPS].sum() / total),
            "tracked runs, median (s)": "{:.1f}".format(np.median(seen) / FPS),
            "longest tracked run (s)": "{:.1f}".format(seen.max() / FPS)})
    gap_stats = pd.DataFrame(rows)

    # E. dips in the player count (<=17 slots tracked) as runs.
    rows = []
    for half, first, last in blocks:
        low = tracked[first:last + 1] <= 17
        r = runs(low)
        rows.append({"half": half, "frames with <=17 slots": pct(low.mean()),
                     "dip runs": len(r),
                     "run length, median (s)": "{:.1f}".format(np.median(r) / FPS) if len(r) else "-",
                     "max (s)": "{:.1f}".format(r.max() / FPS) if len(r) else "-"})
    dip_stats = pd.DataFrame(rows)

    return {"headline": headline, "slot_distribution": slot_distribution,
            "minute_stats": minute_stats, "gap_stats": gap_stats, "dip_stats": dip_stats,
            "per_minute": per_minute}


# Provenance --------------------------------------------------------------------------------------
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(("git", "-C", ROOT) + args, stderr=subprocess.DEVNULL
                                       ).decode().strip()
    except Exception:  # noqa: BLE001 - provenance must not stop the analysis
        return "unavailable"


def provenance(inputs):
    dirty = git("status", "--porcelain", "--untracked-files=no")
    lines = [
        "- generated (UTC): {}".format(datetime.datetime.now(datetime.timezone.utc)
                                      .strftime("%Y-%m-%d %H:%M:%S")),
        "- command: `python {}`".format(" ".join(sys.argv)),
        "- git commit: `{}`{}".format(git("rev-parse", "HEAD"),
                                      " (tracked files had uncommitted changes)" if dirty else ""),
        "- script sha256: `{}`".format(sha256(os.path.abspath(__file__))),
        "- python {} / numpy {} / pandas {}".format(platform.python_version(), np.__version__,
                                                    pd.__version__),
    ]
    for label, path in inputs:
        lines.append("- input {}: `{}`  sha256 `{}`".format(
            label, os.path.relpath(path, ROOT), sha256(path)))
    return "\n".join(lines)


# Output ------------------------------------------------------------------------------------------
def write_markdown(tables, prov, path):
    text = """# Missingness of the tracking on the scored frames

Generated by `explore/missingness_summary.py`. Every number below is computed from the tracking
text only (no ground truth). Definitions are in the script's docstring: *graded* = the two halves;
*tracked slot* = a track slot with a position; *gap* = a run of consecutive frames with the ball
missing; *minute* = 1,500 graded frames from the start of a half.

## Provenance

{prov}

## 1. Headline rates

{headline}

## 2. How many of the 46 slots are tracked per frame

Share of graded frames by number of tracked slots, and how often the ball is missing in that bin.

{slots}

## 3. Minute-to-minute range

Statistics over full 1,500-frame minutes (the partial last minute of each half is excluded from the
quantiles but listed in the CSV).

{minutes}

## 4. Ball gaps and tracked runs

{gaps}

## 5. Dips in the player count

{dips}
""".format(prov=prov, headline=pipe_table(tables["headline"]),
           slots=pipe_table(tables["slot_distribution"]), minutes=pipe_table(tables["minute_stats"]),
           gaps=pipe_table(tables["gap_stats"]), dips=pipe_table(tables["dip_stats"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


def plot_overview(per_minute, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex="col")
    for col, half in enumerate((1, 2)):
        part = per_minute[per_minute["half"] == half]
        axes[0, col].bar(part["minute"], 100 * part["ball_missing"], color="#d62728", width=0.9)
        axes[0, col].set_ylim(0, 100)
        axes[0, col].set_title("Half {}: ball missing per minute".format(half), fontsize=10)
        axes[1, col].plot(part["minute"], part["tracked"], color="black", marker="o", markersize=3)
        axes[1, col].set_ylim(0, 24)
        axes[1, col].set_title("Half {}: mean tracked slots per minute (of 46)".format(half),
                               fontsize=10)
        axes[1, col].set_xlabel("minute of the half")
    axes[0, 0].set_ylabel("% of frames")
    axes[1, 0].set_ylabel("slots")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-md", default=RESULT_MD)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args(argv)

    paths = asset_paths()
    print("reading tracking ...")
    ball_visible, present = load_presence(paths["atd_raw"])
    slot_team, side_of_team = fe.MatchData._read_teams(paths["atd_meta"])
    side_of_slot = np.array([side_of_team[slot_team[s]] for s in range(1, N_SLOTS + 1)])

    tables = analyse(ball_visible, present, side_of_slot)
    prov = provenance([("tracking", paths["atd_raw"]), ("tracking metadata", paths["atd_meta"])])

    write_markdown(tables, prov, args.out_md)
    os.makedirs(args.out_dir, exist_ok=True)
    tables["per_minute"].to_csv(os.path.join(args.out_dir, "missingness_per_minute.csv"), index=False)
    plot_overview(tables["per_minute"], os.path.join(args.out_dir, "missingness_overview.png"))

    for name in ("headline", "slot_distribution", "minute_stats", "gap_stats", "dip_stats"):
        print("\n== {} ==".format(name))
        print(tables[name].to_string(index=False))
    print("\nwrote {}".format(os.path.relpath(args.out_md, ROOT)))


if __name__ == "__main__":
    main()
