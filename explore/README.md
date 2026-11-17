# explore/ - looking at the data one frame at a time

Before building a possession model we need to *see* what the provided data says about a single
moment of the match, and to check that our reading of it is right. This folder does that.

For one video frame (a chosen one, or a random draw) it shows, together:

| Shown | Source |
|---|---|
| the video frame | tactical `.mp4`, decoded with `ffmpeg` |
| tracked players + ball drawn **on the video** | tracking (`_atd.txt`) projected through the homography |
| the same tracking on a **pitch diagram** in metres | tracking |
| ball position, speed, and how long it has been missing | tracking |
| every tracked slot: team, position, speed, distance to ball, pixel position | tracking + metadata XML |
| homography validity and fit error | `_homography.csv` |
| Smart Tagging phases active at the frame | `_smart_tagging.xml` |
| public ground-truth label, if the frame is in the public 5 minutes | `evaluation/gt/..._public.csv` |

### The raw view (`--raw`)

The figure and report are *decoded* data. `--raw` additionally prints, and saves as
`explore/output/frame_<n>_raw.txt`, the untouched source text behind the frame so you can see how
the data actually arrives:

1. **Tracking** - the frame's whole line from `_atd.txt` (`frame_id : 46 slot tokens : ball`),
   wrapped only at `;`, then each tracked slot's raw token next to its decoded metres and team, the
   list of `NaN` slots, and the ball token.
2. **Homography** - the CSV header and the frame's row, the 3x3 matrix, and a worked example
   (`u`, `v`, `w`, then pixel = `u/w, v/w`) for the ball: where its marker lands on the 1920x1080
   video.
3. **Team of each slot** - the `<Player>` entries from the metadata XML (the tracking text itself
   carries no team).
4. **Smart Tagging** - the `<instance>` XML of every phase covering the frame, with the
   seconds-to-frame conversion shown.
5. **Public ground truth** - the CSV rows around the frame.

Every quoted block was compared byte-for-byte with the source file (see the worklog).

## Quick start

Needs the match data in `data/` ([DATA.md](../DATA.md)), `ffmpeg` on the PATH, and Python packages
`numpy`, `pandas`, `matplotlib`, `Pillow`. No `kloppy`, no OpenCV, no Jupyter needed for the script.

```bash
python explore/frame_explorer.py                    # random graded frame
python explore/frame_explorer.py --seed 7           # reproducible random frame
python explore/frame_explorer.py --frame 11500      # a specific video frame
python explore/frame_explorer.py --frame 11500 --raw  # also the RAW source text behind the frame
python explore/frame_explorer.py --ball missing     # random frame where the ball is NOT tracked
python explore/frame_explorer.py --half 2 --context # also save a ball/player-availability plot
python explore/frame_explorer.py --check-seek       # re-verify frame-accurate video seeking
```

The report is printed and the figure is saved to `explore/output/frame_<n>.png`. Add `--show` to
open a window. Start-up takes a few seconds (it reads about 200 MB of tracking text once).

The notebook [01_frame_explorer.ipynb](01_frame_explorer.ipynb) does the same step by step with
explanations (`pip install notebook` to open it; `requirements-starter.txt` lists it).

From Python:

```python
import sys; sys.path.insert(0, "explore")
import frame_explorer as fe
data = fe.MatchData()                       # load once
info = fe.explore(data, seed=42)            # figure + printed report; returns the pieces
info["info"]["players"]                     # DataFrame of tracked slots for that frame
data.ball(11500), data.tagging(11500)       # single queries, no plotting
```

## Files

| File | Purpose |
|---|---|
| `frame_explorer.py` | the module and command-line tool. Reuses `games.py`, `starter/load_homography.py`, `starter/load_smart_tagging.py` and `starter/utils.draw_pitch` rather than re-implementing them |
| `01_frame_explorer.ipynb` | guided notebook over the same functions |
| `missingness_summary.py` | how much of the ball and player tracking is missing on the scored frames, per half and per minute, with gap lengths. Writes `docs/results/missingness_summary.md` with a provenance block (`python explore/missingness_summary.py`) |
| `output/` | rendered images and per-minute CSVs. **Git-ignored on purpose** (see Rules) |

## How the data lines up (as used by the code)

| Item | Convention | Where it is applied |
|---|---|---|
| video frame | index into the tactical video, from 0 = row of a submission | everywhere |
| tracking | line `i` of `_atd.txt` holds `frame_id = i + 1`; the code asserts this on every read | `MatchData.players` |
| coordinates | normalised 0-1; metres = x * 105.3, y * 68.0 | `games.GAMES["BAR-ATH"]["pitch"]` |
| homography | one 3x3 matrix per video frame, maps normalised pitch coords to image pixels | `load_homography.project` |
| Smart Tagging | seconds from video start, frame = round(s * 25), end exclusive | `load_smart_tagging.load` |
| teams | slot to team from the metadata XML; first team listed (Barcelona) = home `H`, second (Athletic) = away `A` | `MatchData._read_teams` |
| video time | `frame / 25` seconds, printed as mm:ss.ff of *video* time (not the match clock) | `_timestamp` |

## What was checked, and how

Each claim below was verified on the real data on 2026-11-11; details and numbers are in
[docs/worklog.md](../docs/worklog.md).

* **The video is constant 25 fps, 1920x1080, 199,319 frames, starting at t = 0** (`ffprobe`), the
  same frame count as the tracking and the submission.
* **Seeking is frame-accurate.** `ffmpeg -ss frame/25` returns pixel-identical images to a
  sequential decode of the same frame (frames 11500 and 20000; `--check-seek`).
* **Tracking, homography and video agree.** Projecting the tracked players through the homography
  lands the markers on the actual players in the video (inspected at frames 11500 and 135406).
* **Coordinates are absolute, and the teams swap ends at half-time.** Slot 24 (Barcelona's
  goalkeeper in most frames) has mean x = 0.10 in the first half and 0.88 in the second. So
  "Barcelona attack left to right" holds only for the first half. Any feature such as "own half"
  or "attacking third" must take the half into account.

## Known quirks

* **The video file may carry a suffix.** Ours is `BAR-ATH_tactical-002.mp4`, not the
  `BAR-ATH_tactical.mp4` that `games.py` builds. `find_video()` accepts exactly one
  `BAR-ATH_tactical*.mp4`, so nothing has to be renamed. (Other repo code that opens the video via
  `games.asset_paths()` would need the exact name.)
* **Slots are not players.** They appear, vanish and swap identity, so slot 24 is the goalkeeper in
  most frames but not all (frame 135406 is a counter-example).
* **`frozen` is a heuristic.** It flags a slot whose position is bit-for-bit identical to the
  previous frame; sampled over the first half this is rare (0-2.4% of a slot's frames; slots 24 and
  46 are highest). It suggests a stale track; it is not proof.
* **Ball speed is derived**, not provided: mean speed over the last 5 frames when the ball is
  tracked now and 5 frames earlier. The raw ball column has no speed value.
* **Line endings differ between files.** The homography CSV and the Smart Tagging XML use `\r\n`;
  the tracking text and the ground-truth CSV use `\n`. Python's text mode, pandas and the XML parser
  hide this; anything comparing raw lines byte-for-byte (shell tools, binary reads) must strip `\r`.
* **Team names are spelled as in the files**, e.g. `Atletic Club` (one `h`).
* **`public GT: n/a`** for almost every frame: the public label covers frames 11185-18684 only.

## Rules

* The public ground-truth label is shown only so that we can sanity-check our reading of the data.
  It is not used for tuning, thresholds or training ([RULES.md](../RULES.md)).
* `explore/output/` is git-ignored. A video frame, or a plot made from it, is derived from the gated
  match data and must not be committed to the public repository. The code here is safe to commit;
  its output is not.
* Nothing in this folder labels anything by hand; it only displays existing data.

## Extending it

Add a query as a method on `MatchData`, include it in `build_report` and `format_report`, and add a
panel in `make_figure` if it is visual. Keep the one-sentence "why" docstring style, and record
anything learned in `docs/worklog.md`.
