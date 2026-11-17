"""Frame explorer: look at one video frame together with every piece of reference data about it.

Why this exists
---------------
Before building anything, we need to *see* what the provided data says about a single moment of the
match. This module takes a video frame number (or draws a random one) and shows, side by side:

  * the video frame itself (decoded straight from the tactical .mp4 with ffmpeg),
  * the tracked players and the ball projected onto that frame through the homography, so we can
    check that the tracking and the video really describe the same moment,
  * the same tracking drawn on a pitch diagram, in metres,
  * a printed report: ball position/speed/missing-gap, every tracked player, homography quality,
    the Smart Tagging phases active at that frame, and the public ground-truth label if the frame
    falls inside the public 5 minutes.

Usage
-----
    python explore/frame_explorer.py                   # random graded frame, saved to explore/output/
    python explore/frame_explorer.py --seed 7          # same random frame every time
    python explore/frame_explorer.py --frame 11500     # a specific video frame
    python explore/frame_explorer.py --ball missing    # random frame where the ball is NOT tracked
    python explore/frame_explorer.py --help

or from a notebook / Python:

    import frame_explorer as fe
    data = fe.MatchData()                  # loads tracking, homography, tagging once (~seconds)
    info = fe.explore(data, seed=42)       # draws the figure, prints the report, returns the pieces

Conventions (all of them are stated in DATA.md; they are repeated here because getting one wrong
silently shifts every frame)
-----------------------------------------------------------------------------------------------
  * "video frame" = row of a submission = index into the tactical video, counting from 0.
  * The tracking record for a video frame has ``frame_id = video_frame + 1``.
  * Tracking coordinates are normalised 0-1; metres = x * 105.3, y * 68.0 (games.py).
  * The homography maps normalised pitch coordinates (0-1) to image pixels, one matrix per frame.
  * Smart Tagging intervals are in seconds from the first video frame; frame = round(seconds*25).
  * Tracking slots are NOT players: 46 slots, they appear/vanish/swap identity. Team membership of
    each slot comes from the metadata XML.

What this module does NOT do
----------------------------
  * It never uses the ground truth for anything except printing it next to the frame, and it only
    has the public 5 minutes. That label is for sanity-checking our reading of the data -- it must
    not be used to tune or train anything (RULES.md).
  * It writes images to ``explore/output/``, which is git-ignored: a video frame is derived from
    the gated match data and must not end up in the public repository.
"""

import argparse
import glob
import io
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths, half_of_video_frame, half_video_blocks  # noqa: E402
from starter import load_homography, load_smart_tagging  # noqa: E402
from starter.utils import draw_pitch  # noqa: E402

# Params ----------------------------------------------------------------------------------------
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH_M, PITCH_WIDTH_M = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
IMAGE_SIZE = (1920, 1080)  # width, height of the tactical video

PUBLIC_GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUTPUT_DIR = os.path.join(HERE, "output")

# Same team colours as the project document: Barcelona (home, H) blue, Athletic (away, A) orange.
COLOUR = {"H": "#1f77b4", "A": "#ff7f0e"}
BALL_COLOUR = "#ffe600"


# Locating the video ----------------------------------------------------------------------------
def find_video(data_dir=None):
    """Path of the tactical video.

    DATA.md says the file is called ``BAR-ATH_tactical.mp4``, but a browser/download manager may
    add a suffix (our copy is ``BAR-ATH_tactical-002.mp4``). Rather than make everyone rename a
    5 GB file, accept exactly one ``BAR-ATH_tactical*.mp4`` in the tactical folder.
    """
    expected = asset_paths(data_dir=data_dir)["video"]
    if os.path.isfile(expected):
        return expected
    hits = sorted(glob.glob(os.path.join(os.path.dirname(expected), "BAR-ATH_tactical*.mp4")))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise IOError("tactical video not found. Expected {} (or BAR-ATH_tactical*.mp4 next to "
                      "it). See DATA.md.".format(expected))
    raise IOError("several candidate videos, keep one (or fix the name): {}".format(hits))


def _tidy_xml(text):
    """One XML element as readable text: drop the file's deep indentation, keep its structure."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) > 1:
        indent = min(len(l) - len(l.lstrip()) for l in lines[1:])
        lines = [lines[0]] + [l[indent:] for l in lines[1:]]
    return "\n".join(lines)


def _timestamp(video_frame):
    """mm:ss.ff for a video frame, for display only."""
    seconds = video_frame / float(FPS)
    return "{:02d}:{:05.2f}".format(int(seconds // 60), seconds % 60)


# The data --------------------------------------------------------------------------------------
class MatchData(object):
    """Everything provided for the tactical recording, loaded once and queried per frame.

    Loading reads the whole raw tracking text (about 200 MB) and keeps only the ball column as an
    array; player columns are parsed on demand for the one frame being looked at, which keeps
    start-up to a few seconds.
    """

    def __init__(self, data_dir=None):
        paths = asset_paths(data_dir=data_dir)
        self.paths = paths
        self.video_path = find_video(data_dir)

        # Tracking: raw lines, one per video frame. Line i holds frame_id i + 1.
        with open(paths["atd_raw"], "r") as handle:
            self._lines = handle.read().splitlines()
        if len(self._lines) != N_FRAMES:
            raise ValueError("expected {} tracking lines, found {}".format(
                N_FRAMES, len(self._lines)))

        # Ball column for every frame as an (N, 2) array of normalised x, y; NaN where untracked.
        ball = np.empty((N_FRAMES, 2))
        for i, line in enumerate(self._lines):
            fields = line.rsplit(":", 1)[1].split(",")
            ball[i, 0], ball[i, 1] = float(fields[0]), float(fields[1])
        self.ball_xy = ball
        self.ball_visible = ~np.isnan(ball[:, 0])
        index = np.arange(N_FRAMES)
        # For every frame: the nearest earlier / later frame in which the ball IS tracked.
        self._last_seen = np.maximum.accumulate(np.where(self.ball_visible, index, -1))
        self._next_seen = np.minimum.accumulate(
            np.where(self.ball_visible, index, 2 * N_FRAMES)[::-1])[::-1]

        self.slot_team, self.side_of_team = self._read_teams(paths["atd_meta"])

        self.homography, self.reproj_error, self.h_valid = load_homography.load()
        self.tags = load_smart_tagging.load()
        # The XML elements behind self.tags, in the same order (both iterate <instance> in file
        # order), kept so the raw view can show the untouched source text of a phase.
        self._tag_elements = list(ET.parse(paths["smart_tagging"]).getroot().iter("instance"))
        self.gt = pd.read_csv(PUBLIC_GT_PATH)["label"].values

    @staticmethod
    def _read_teams(meta_path):
        """({slot: team name}, {team name: 'H' or 'A'}) from the tracking metadata.

        The first team listed in the metadata is Barcelona, which DATA.md names as the home side
        (``H``); the second is Athletic (``A``).
        """
        root = ET.parse(meta_path).getroot()
        names = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
        name_of_id = dict(names)
        slot_team = {int(p.get("id")): name_of_id[p.get("teamId")] for p in root.iter("Player")}
        side = {names[0][1]: "H", names[1][1]: "A"}
        return slot_team, side

    # -- per-frame queries ---------------------------------------------------------------------
    def frame_info(self, video_frame):
        """Where this frame sits in the match: time, half, whether it is graded."""
        half = half_of_video_frame(GAME, video_frame)
        return {
            "video_frame": video_frame,
            "frame_id": video_frame + 1,
            "seconds": video_frame / float(FPS),
            "timestamp": _timestamp(video_frame),
            "half": half,
            "graded": half is not None,
        }

    def public_label(self, video_frame):
        """Public ground-truth label (H/A/D), or None when the frame is not in the public 5 min."""
        label = self.gt[video_frame]
        return None if label == "X" else label

    def ball(self, video_frame, speed_window=5):
        """Ball status at a frame: position, speed, and how long it has been / will be missing."""
        visible = bool(self.ball_visible[video_frame])
        x, y = self.ball_xy[video_frame]
        last, nxt = self._last_seen[video_frame], self._next_seen[video_frame]
        before = video_frame - speed_window
        speed = None
        if visible and before >= 0 and self.ball_visible[before]:
            dx = (x - self.ball_xy[before, 0]) * PITCH_LENGTH_M
            dy = (y - self.ball_xy[before, 1]) * PITCH_WIDTH_M
            speed = float(np.hypot(dx, dy)) / (speed_window / float(FPS))
        return {
            "visible": visible,
            "x": x, "y": y,
            "x_m": x * PITCH_LENGTH_M, "y_m": y * PITCH_WIDTH_M,
            # Mean speed over the last `speed_window` frames. The raw file has no ball speed.
            "speed_ms": speed,
            "frames_since_seen": None if last < 0 else int(video_frame - last),
            "frames_until_seen": None if nxt >= N_FRAMES else int(nxt - video_frame),
        }

    def players(self, video_frame):
        """One row per tracked slot that has a position at this frame.

        Columns: slot, team, side (H/A), x, y (normalised), x_m, y_m, speed_ms (from the file,
        NaN when absent), dist_ball_m (NaN if the ball is not tracked), px, py (image pixels via
        the homography, NaN if it is invalid) and ``frozen``: the slot's position is bit-for-bit
        identical to the previous frame. A moving player essentially never does that, so a frozen
        slot is possibly a stale track that kept its last value. It is a heuristic flag, not ground
        truth: sampled over the first half it is rare (about 0-2% of a slot's frames).
        """
        frame_id, slots, _ = self._lines[video_frame].split(":")
        if int(frame_id) != video_frame + 1:
            raise ValueError("tracking line {} holds frame_id {}, expected {}".format(
                video_frame, frame_id, video_frame + 1))
        previous = (self._lines[video_frame - 1].split(":")[1].split(";")
                    if video_frame > 0 else None)

        rows = []
        for index, token in enumerate(slots.split(";")):
            if token.startswith("NaN"):
                continue
            x_text, y_text, speed_text = token.split(",")
            slot = index + 1
            team = self.slot_team[slot]
            frozen = previous is not None and previous[index].split(",")[:2] == [x_text, y_text]
            rows.append({
                "slot": slot, "team": team, "side": self.side_of_team[team],
                "x": float(x_text), "y": float(y_text), "speed_ms": float(speed_text),
                "frozen": frozen,
            })
        table = pd.DataFrame(rows, columns=["slot", "team", "side", "x", "y", "speed_ms", "frozen"])
        table["x_m"] = table["x"] * PITCH_LENGTH_M
        table["y_m"] = table["y"] * PITCH_WIDTH_M

        ball = self.ball(video_frame)
        if ball["visible"]:
            table["dist_ball_m"] = np.hypot(table["x_m"] - ball["x_m"], table["y_m"] - ball["y_m"])
        else:
            table["dist_ball_m"] = np.nan

        if self.h_valid[video_frame]:
            px, py = load_homography.project(self.homography[video_frame], table["x"], table["y"])
            table["px"], table["py"] = px, py
        else:
            table["px"] = table["py"] = np.nan
        return table

    def homography_info(self, video_frame):
        """Whether a matrix exists for the frame and how far its fit misses (lower is better)."""
        return {"valid": bool(self.h_valid[video_frame]),
                "reprojection_error": float(self.reproj_error[video_frame])}

    def tagging(self, video_frame):
        """Smart Tagging phases whose interval contains the frame (end is exclusive)."""
        return [event for event in self.tags
                if event["start_frame"] <= video_frame < event["end_frame"]]

    def image(self, video_frame):
        """The video frame as an (H, W, 3) uint8 RGB array, decoded with ffmpeg.

        ffmpeg seeks to the timestamp ``frame / 25`` and decodes exactly one frame. The video is
        constant 25 fps with 199,319 frames starting at t = 0 (checked with ffprobe), so this is
        frame-accurate; ``python explore/frame_explorer.py --check-seek`` re-verifies that against
        a sequential decode.
        """
        from PIL import Image

        command = ["ffmpeg", "-v", "error", "-ss", "{:.3f}".format(video_frame / float(FPS)),
                   "-i", self.video_path, "-frames:v", "1", "-f", "image2pipe",
                   "-vcodec", "png", "-"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError("ffmpeg could not decode frame {}: {}".format(
                video_frame, result.stderr.decode("utf-8", "replace")))
        return np.asarray(Image.open(io.BytesIO(result.stdout)).convert("RGB"))

    def raw(self, video_frame, slots=()):
        """The untouched source text behind one frame, straight from the files.

        Returns strings only, nothing decoded: the tracking line, the homography CSV row (with its
        header), the public ground-truth CSV rows around the frame, the XML of every Smart Tagging
        phase covering the frame, and the metadata XML entries of the given track ``slots``.
        """
        # Homography CSV: header is line 0, video frame f is line f + 1.
        with open(self.paths["homography"], "r") as handle:
            for number, line in enumerate(handle):
                if number == 0:
                    homography_header = line.rstrip("\n")
                if number == video_frame + 1:
                    homography_row = line.rstrip("\n")
                    break

        with open(PUBLIC_GT_PATH, "r") as handle:
            gt_lines = handle.read().splitlines()  # header + one row per frame, same indexing
        gt_rows = {f: gt_lines[f + 1] for f in range(video_frame - 1, video_frame + 2)
                   if 0 <= f < N_FRAMES}

        covering = [i for i, event in enumerate(self.tags)
                    if event["start_frame"] <= video_frame < event["end_frame"]]
        tag_xml = [(self.tags[i], _tidy_xml(ET.tostring(self._tag_elements[i], encoding="unicode")))
                   for i in covering]

        meta_root = ET.parse(self.paths["atd_meta"]).getroot()
        team_name = {t.get("id"): t.findtext("Name") for t in meta_root.iter("Team")}
        meta_players = []
        for player in meta_root.iter("Player"):
            if int(player.get("id")) in set(slots):
                meta_players.append(
                    '<Player id="{}" teamId="{}">  Name={}  ShirtNumber={}   -> team "{}"'.format(
                        player.get("id"), player.get("teamId"), player.findtext("Name"),
                        player.findtext("ShirtNumber"), team_name[player.get("teamId")]))

        return {
            "tracking_line": self._lines[video_frame],
            "homography_header": homography_header,
            "homography_row": homography_row,
            "gt_header": gt_lines[0],
            "gt_rows": gt_rows,
            "tag_xml": tag_xml,
            "meta_players": meta_players,
        }

    def players_seen(self, start, stop):
        """Number of tracked slots with a position for each frame in [start, stop)."""
        return np.array([sum(1 for token in self._lines[i].split(":")[1].split(";")
                             if not token.startswith("NaN")) for i in range(start, stop)])

    # -- choosing a frame ----------------------------------------------------------------------
    def graded_frames(self, half=None):
        """Video frames that are scored (the two halves), optionally only one half."""
        blocks = [b for b in half_video_blocks(GAME) if half in (None, b[0])]
        return np.concatenate([np.arange(first, last + 1) for _, first, last in blocks])

    def sample_frame(self, seed=None, ball="any", half=None, graded_only=True):
        """A random video frame. ``ball`` is 'any', 'present' or 'missing' (is the ball tracked).

        ``seed`` makes the draw reproducible, so a frame can be quoted in a report and re-found.
        """
        rng = np.random.RandomState(seed)
        pool = self.graded_frames(half) if graded_only else np.arange(N_FRAMES)
        if ball == "present":
            pool = pool[self.ball_visible[pool]]
        elif ball == "missing":
            pool = pool[~self.ball_visible[pool]]
        elif ball != "any":
            raise ValueError("ball must be 'any', 'present' or 'missing'")
        return int(rng.choice(pool))


# Reporting -------------------------------------------------------------------------------------
def build_report(data, video_frame):
    """Everything known about one frame, as a dict of the raw pieces (no printing, no plotting)."""
    return {
        "frame": data.frame_info(video_frame),
        "public_label": data.public_label(video_frame),
        "ball": data.ball(video_frame),
        "players": data.players(video_frame),
        "homography": data.homography_info(video_frame),
        "tags": data.tagging(video_frame),
    }


def format_report(info):
    """The report as text. Plain strings so it reads the same in a terminal and a notebook."""
    frame, ball, players = info["frame"], info["ball"], info["players"]
    out = []
    out.append("=" * 78)
    out.append("VIDEO FRAME {video_frame}   (tracking frame_id {frame_id})   t = {timestamp}"
               .format(**frame))
    where = "half {}, graded".format(frame["half"]) if frame["graded"] else "NOT graded (outside the two halves)"
    out.append("position in match : {}".format(where))
    label = info["public_label"]
    out.append("public ground truth: {}".format(
        label if label else "none -- frame is outside the public 5 minutes (11185..18684)")
        + ("   [for sanity checks only, never for tuning]" if label else ""))

    out.append("-" * 78)
    if ball["visible"]:
        speed = "n/a" if ball["speed_ms"] is None else "{:.1f} m/s (last 5 frames)".format(ball["speed_ms"])
        out.append("BALL tracked      : x={x:.4f} y={y:.4f}  =  ({x_m:.1f} m, {y_m:.1f} m)".format(**ball))
        out.append("ball speed        : " + speed)
    else:
        out.append("BALL NOT TRACKED at this frame")
        seen_before = ball["frames_since_seen"]
        seen_after = ball["frames_until_seen"]
        out.append("last seen         : {}".format(
            "never before" if seen_before is None else "{} frames ago ({:.1f} s)".format(
                seen_before, seen_before / float(FPS))))
        out.append("next seen         : {}".format(
            "never again" if seen_after is None else "in {} frames ({:.1f} s)".format(
                seen_after, seen_after / float(FPS))))

    homography = info["homography"]
    out.append("homography        : valid={valid}  reprojection_error={reprojection_error:.3f}"
               "  (lower is better; median in play is about 0.24)".format(**homography))

    out.append("-" * 78)
    out.append("TRACKED SLOTS: {} of 46 have a position  (Barcelona {}, Athletic {})".format(
        len(players), int((players["side"] == "H").sum()), int((players["side"] == "A").sum())))
    columns = ["slot", "side", "x_m", "y_m", "speed_ms", "dist_ball_m", "px", "py", "frozen"]
    shown = players[columns].sort_values("dist_ball_m" if ball["visible"] else "slot")
    out.append(shown.round(1).to_string(index=False))
    if ball["visible"] and len(players):
        nearest = players.loc[players["dist_ball_m"].idxmin()]
        out.append("nearest slot to the ball: {} ({}), {:.1f} m away".format(
            int(nearest["slot"]), nearest["team"], nearest["dist_ball_m"]))
    if players["frozen"].any():
        out.append("frozen slots (same position as the previous frame, possibly stale tracks): {}"
                   .format(sorted(players.loc[players["frozen"], "slot"].astype(int).tolist())))

    out.append("-" * 78)
    if info["tags"]:
        out.append("SMART TAGGING phases active at this frame:")
        for event in info["tags"]:
            out.append("  {code:<30} frames {start_frame}..{end_frame}   tags: {tags}".format(**event))
    else:
        out.append("SMART TAGGING: no phase covers this frame")
    out.append("=" * 78)
    return "\n".join(out)


def _wrap_tokens(prefix, tokens, width=100):
    """Lay out ';'-separated tokens over several lines without cutting any token in two."""
    lines, current = [], prefix
    for i, token in enumerate(tokens):
        piece = token + (";" if i < len(tokens) - 1 else "")
        if len(current) + len(piece) > width and current.strip():
            lines.append(current)
            current = "    "
        current += piece
    lines.append(current)
    return lines


def format_raw(data, video_frame):
    """The raw source text behind one frame, step by step, with how each part becomes the figure.

    Nothing here is computed from the data: every block quoted from a file is copied verbatim; the
    "->" notes and the worked homography example are derived from it so the conversion can be
    followed by hand.
    """
    info = build_report(data, video_frame)
    players, ball = info["players"], info["ball"]
    raw = data.raw(video_frame, slots=players["slot"].astype(int).tolist())
    rel = lambda path: os.path.relpath(path, ROOT)  # noqa: E731
    out, bar = [], "=" * 78

    out += [bar, "RAW SOURCE DATA FOR VIDEO FRAME {}".format(video_frame),
            "Text in the [1]..[5] blocks is copied from the files; '->' lines explain the step to "
            "the figure.", bar, ""]

    # [1] tracking ---------------------------------------------------------------------------
    frame_id, slot_part, ball_part = raw["tracking_line"].split(":")
    tokens = slot_part.split(";")
    out += ["[1] TRACKING   {}".format(rel(data.paths["atd_raw"])),
            "    line {} of the file (1-based) holds frame_id {}  ->  video frame = frame_id - 1 = {}"
            .format(video_frame + 1, frame_id, video_frame),
            "    one line per frame:   frame_id : slot1 ; slot2 ; ... ; slot46 : ball",
            "    every slot / ball token is   x,y,speed   (x,y normalised 0-1, speed m/s, NaN = missing)",
            "", "    the raw line ({} characters), wrapped only at ';':".format(len(raw["tracking_line"]))]
    out += _wrap_tokens("    " + frame_id + ":", tokens)
    out.append("    :" + ball_part)

    tracked = [(i + 1, t) for i, t in enumerate(tokens) if not t.startswith("NaN")]
    missing = [i + 1 for i, t in enumerate(tokens) if t.startswith("NaN")]
    out += ["", "    decoded, slot by slot ({} of 46 slots have a position):".format(len(tracked)),
            "      slot  raw token                                       x_m     y_m   team"]
    for slot, token in tracked:
        row = players[players["slot"] == slot].iloc[0]
        out.append("      {:>4}  {:<44} {:>6.1f}  {:>6.1f}   {}".format(
            slot, token, row["x_m"], row["y_m"], row["team"]))
    out.append("      NaN,NaN,NaN (not tracked) in slots: {}".format(missing))
    ball_x, ball_y = ball_part.split(",")[:2]
    out += ["", "    ball token: {}".format(ball_part)]
    if ball["visible"]:
        out.append("      -> x = {} * 105.3 = {:.2f} m,  y = {} * 68.0 = {:.2f} m.  The third value is "
                   "NaN: the file carries no ball speed.".format(
                       ball_x, ball["x_m"], ball_y, ball["y_m"]))
    else:
        out.append("      -> NaN,NaN,NaN: the ball is not tracked in this frame.")

    # [2] homography -------------------------------------------------------------------------
    out += ["", "[2] HOMOGRAPHY   {}".format(rel(data.paths["homography"])),
            "    row {} of the file, after the header (one row per video frame, counting from 0):"
            .format(video_frame),
            "    " + raw["homography_header"], "    " + raw["homography_row"]]
    matrix = data.homography[video_frame]
    if data.h_valid[video_frame]:
        out += ["", "    h00..h22 read row by row as the 3x3 matrix H (maps normalised pitch x,y to pixels):"]
        out += ["        [{:>11.4f} {:>11.4f} {:>11.4f}]".format(*matrix[r]) for r in range(3)]
        # Worked example on the ball if tracked, else on the first tracked slot.
        if ball["visible"]:
            label, x, y = "ball", ball["x"], ball["y"]
        else:
            first = players.iloc[0]
            label, x, y = "slot {}".format(int(first["slot"])), first["x"], first["y"]
        u = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        v = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        px, py = load_homography.project(matrix, x, y)
        out += ["", "    worked example for the {} (x={:.5f}, y={:.5f}):".format(label, x, y),
                "      u = h00*x + h01*y + h02 = {:.2f}".format(u),
                "      v = h10*x + h11*y + h12 = {:.2f}".format(v),
                "      w = h20*x + h21*y + h22 = {:.4f}".format(w),
                "      pixel = (u/w, v/w) = ({:.1f}, {:.1f})   <- where the {} marker is drawn on "
                "the 1920x1080 video".format(px, py, label)]
    else:
        out += ["", "    valid = 0: the matrix columns are NaN, so nothing can be projected onto the video."]

    # [3] team metadata ----------------------------------------------------------------------
    out += ["", "[3] TEAM OF EACH SLOT   {}".format(rel(data.paths["atd_meta"])),
            "    the tracking text has no team; the metadata XML names a team for every slot "
            "(shown for the tracked slots):"]
    out += ["      " + line for line in raw["meta_players"]]
    out.append("    -> first team listed (Barcelona) = home H = blue, second (Athletic) = away A = orange.")

    # [4] smart tagging ----------------------------------------------------------------------
    out += ["", "[4] SMART TAGGING   {}".format(rel(data.paths["smart_tagging"]))]
    if raw["tag_xml"]:
        out.append("    <instance> blocks whose interval covers this frame (times are seconds from "
                   "the first video frame):")
        for event, xml_text in raw["tag_xml"]:
            out += ["", "      " + xml_text.replace("\n", "\n      "),
                    "      -> start {} s * 25 = frame {},  end {} s * 25 = frame {}  (end exclusive); "
                    "{} <= {} < {}".format(event["start_s"], event["start_frame"], event["end_s"],
                                           event["end_frame"], event["start_frame"], video_frame,
                                           event["end_frame"])]
    else:
        out.append("    no <instance> covers this frame.")

    # [5] ground truth -----------------------------------------------------------------------
    out += ["", "[5] PUBLIC GROUND TRUTH   {}".format(rel(PUBLIC_GT_PATH)),
            "    rows around the frame (header first; one row per video frame):", "      " + raw["gt_header"]]
    for f, line in sorted(raw["gt_rows"].items()):
        out.append("      {}{}".format(line, "     <- this frame" if f == video_frame else ""))
    out.append("    -> H / A / D = who has the ball or dead ball; X = frame not covered by the public "
               "ground truth (only 11185..18684 is).")

    out += ["", "HOW THE RAW VALUES BECOME THE FIGURE",
            "  tracking x,y (0-1) --x105.3, x68.0-------> metres        (right-hand pitch panel)",
            "  tracking x,y (0-1) --homography [2]------> pixels        (markers on the video)",
            "  slot number        --metadata [3]--------> team, colour",
            "  ball and slot metres --Euclidean distance-> dist_ball_m  (nearest-player rule)",
            bar]
    return "\n".join(out)


# Plotting --------------------------------------------------------------------------------------
def make_figure(data, video_frame, info, image=None):
    """Left: the video frame with projected tracking. Right: the same tracking on a pitch."""
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    frame, ball, players = info["frame"], info["ball"], info["players"]
    if image is None:
        image = data.image(video_frame)

    fig = plt.figure(figsize=(17, 6.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.06, top=0.88, bottom=0.04)
    ax_image, ax_pitch = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    ax_image.set_anchor("N")
    ax_pitch.set_anchor("N")

    # -- video frame with the tracking projected on top ---------------------------------------
    ax_image.imshow(image)
    outline = [patheffects.withStroke(linewidth=2.5, foreground="black")]
    if info["homography"]["valid"]:
        corners = load_homography.pitch_corners(data.homography[video_frame])
        xs = [c[0] for c in corners] + [corners[0][0]]
        ys = [c[1] for c in corners] + [corners[0][1]]
        ax_image.plot(xs, ys, "--", color="cyan", linewidth=1, label="pitch edge (homography)")
        for side in ("H", "A"):
            rows = players[players["side"] == side]
            ax_image.scatter(rows["px"], rows["py"], s=70, color=COLOUR[side], edgecolors="white",
                             linewidths=1, zorder=3,
                             label="Barcelona (H)" if side == "H" else "Athletic (A)")
        for _, row in players.iterrows():
            ax_image.text(row["px"] + 8, row["py"] - 8, str(int(row["slot"])), color="white",
                          fontsize=8, path_effects=outline, zorder=4)
        if ball["visible"]:
            bx, by = load_homography.project(data.homography[video_frame], ball["x"], ball["y"])
            ax_image.scatter([bx], [by], s=260, facecolors="none", edgecolors=BALL_COLOUR,
                             linewidths=2.5, zorder=5, label="ball (tracking)")
        ax_image.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax_image.set_xlim(0, IMAGE_SIZE[0])
    ax_image.set_ylim(IMAGE_SIZE[1], 0)
    ax_image.axis("off")
    ax_image.set_title("video frame with tracking projected through the homography "
                       "(numbers are track slots)", fontsize=10)

    # -- pitch diagram -------------------------------------------------------------------------
    draw_pitch(ax_pitch, length=PITCH_LENGTH_M, width=PITCH_WIDTH_M)
    for side in ("H", "A"):
        rows = players[players["side"] == side]
        ax_pitch.scatter(rows["x_m"], rows["y_m"], s=90, color=COLOUR[side], edgecolors="black",
                         linewidths=0.6, zorder=3)
    for _, row in players.iterrows():
        ax_pitch.text(row["x_m"] + 0.8, row["y_m"] - 0.8, str(int(row["slot"])), fontsize=7,
                      zorder=4, color="black", path_effects=[
                          patheffects.withStroke(linewidth=2, foreground="white")])
    if ball["visible"]:
        ax_pitch.scatter([ball["x_m"]], [ball["y_m"]], s=220, facecolors="none",
                         edgecolors=BALL_COLOUR, linewidths=2.5, zorder=5)
        ax_pitch.scatter([ball["x_m"]], [ball["y_m"]], s=22, color="black", zorder=6)
    # y = 0 at the top, so the diagram has the same orientation as the camera view.
    ax_pitch.set_ylim(PITCH_WIDTH_M + 3, -3)
    ax_pitch.set_title("tracking in metres (camera orientation: y = 0 is the far touchline)",
                       fontsize=10)

    label = info["public_label"] or "n/a"
    gap = "" if ball["visible"] else "   |   BALL NOT TRACKED"
    fig.suptitle("video frame {video_frame}   t = {timestamp}   half {half}".format(**frame)
                 .replace("half None", "outside play")
                 + "   |   public GT: {}{}".format(label, gap), fontsize=13, y=0.97)
    return fig


def plot_context(data, video_frame, seconds=20):
    """Is the ball tracked, and how many slots are tracked, in the seconds around this frame?

    Missingness is the central difficulty of the data; this shows whether the ball goes missing
    on its own or together with the players (a tracker outage).
    """
    import matplotlib.pyplot as plt

    start = max(0, video_frame - seconds * FPS)
    stop = min(N_FRAMES, video_frame + seconds * FPS)
    index = np.arange(start, stop)
    t = (index - video_frame) / float(FPS)

    fig, (ax_ball, ax_players) = plt.subplots(2, 1, figsize=(12, 3.8), sharex=True,
                                              gridspec_kw={"height_ratios": [1, 2]})
    ax_ball.fill_between(t, 0, 1, where=data.ball_visible[start:stop], step="mid",
                         color="#2ca02c", label="ball tracked")
    ax_ball.fill_between(t, 0, 1, where=~data.ball_visible[start:stop], step="mid",
                         color="#d62728", label="ball missing")
    ax_ball.set_yticks([])
    ax_ball.legend(loc="upper right", ncol=2, fontsize=8)
    ax_players.plot(t, data.players_seen(start, stop), color="black", linewidth=1)
    ax_players.set_ylabel("slots with\na position")
    ax_players.set_xlabel("seconds from this frame")
    for ax in (ax_ball, ax_players):
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
    fig.suptitle("ball and player tracking around video frame {}".format(video_frame), fontsize=11)
    fig.tight_layout()
    return fig


# Orchestration ---------------------------------------------------------------------------------
def explore(data, frame=None, seed=None, ball="any", half=None, show=True, save=True,
            output_dir=OUTPUT_DIR, quiet=False, raw=False):
    """Pick a frame, draw the figure, print the report, return the pieces.

    ``frame`` wins over ``seed``. Without either, the draw is random each time (the frame number
    is always printed so a good example can be re-found). ``raw=True`` also prints the untouched
    source text behind the frame and saves it next to the figure as ``frame_<n>_raw.txt``.
    """
    import matplotlib.pyplot as plt

    if frame is None:
        frame = data.sample_frame(seed=seed, ball=ball, half=half)
    if not 0 <= frame < N_FRAMES:
        raise ValueError("video frame must be in 0..{}".format(N_FRAMES - 1))

    info = build_report(data, frame)
    image = data.image(frame)
    fig = make_figure(data, frame, info, image=image)

    saved = None
    if save:
        os.makedirs(output_dir, exist_ok=True)
        saved = os.path.join(output_dir, "frame_{:06d}.png".format(frame))
        fig.savefig(saved, dpi=110, bbox_inches="tight")

    text = format_report(info)
    raw_text = format_raw(data, frame) if raw else None
    if not quiet:
        print(text)
        if saved:
            print("figure saved to {}".format(os.path.relpath(saved, ROOT)))
        if raw_text:
            print()
            print(raw_text)
    if raw_text and save:
        raw_path = os.path.join(output_dir, "frame_{:06d}_raw.txt".format(frame))
        with open(raw_path, "w") as handle:
            handle.write(raw_text + "\n")
        if not quiet:
            print("raw view saved to {}".format(os.path.relpath(raw_path, ROOT)))
    if show:
        plt.show()
    return {"frame": frame, "info": info, "image": image, "figure": fig, "text": text,
            "raw_text": raw_text, "saved": saved}


def check_seek(data, frames=(11500, 20000)):
    """Verify ffmpeg's fast seek returns the same pixels as a sequential decode of that frame.

    The reference decode starts at frame 0, so a late frame takes minutes; the default frames are
    early on purpose. It is a one-off check, not something to run per frame.
    """
    ok = True
    for frame in frames:
        fast = data.image(frame)
        command = ["ffmpeg", "-v", "error", "-i", data.video_path, "-vf",
                   "select=eq(n\\,{})".format(frame), "-vsync", "0", "-frames:v", "1",
                   "-f", "image2pipe", "-vcodec", "png", "-"]
        from PIL import Image
        slow = np.asarray(Image.open(io.BytesIO(
            subprocess.run(command, stdout=subprocess.PIPE).stdout)).convert("RGB"))
        same = fast.shape == slow.shape and np.array_equal(fast, slow)
        print("frame {:>7}: fast seek == sequential decode: {}".format(frame, same))
        ok = ok and same
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Show one video frame with all its reference data (tracking, homography, "
                    "Smart Tagging, public ground truth).")
    parser.add_argument("--frame", type=int, help="video frame to show (0..199318)")
    parser.add_argument("--seed", type=int, help="seed for the random draw (reproducible)")
    parser.add_argument("--ball", choices=["any", "present", "missing"], default="any",
                        help="restrict the random draw by whether the ball is tracked")
    parser.add_argument("--half", type=int, choices=[1, 2], help="restrict the draw to one half")
    parser.add_argument("--raw", action="store_true",
                        help="also print the raw source text behind the frame (tracking line, "
                             "homography row, Smart Tagging XML, ground-truth rows)")
    parser.add_argument("--context", action="store_true",
                        help="also save the ball/players-around-this-frame plot")
    parser.add_argument("--show", action="store_true", help="open a window instead of only saving")
    parser.add_argument("--out", default=OUTPUT_DIR, help="output folder (default explore/output)")
    parser.add_argument("--check-seek", action="store_true",
                        help="verify frame-accurate ffmpeg seeking, then exit")
    args = parser.parse_args(argv)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    print("loading tracking, homography and Smart Tagging ...")
    data = MatchData()
    if args.check_seek:
        sys.exit(0 if check_seek(data) else 1)

    result = explore(data, frame=args.frame, seed=args.seed, ball=args.ball, half=args.half,
                     show=args.show, output_dir=args.out, raw=args.raw)
    if args.context:
        import matplotlib.pyplot as plt
        fig = plot_context(data, result["frame"])
        path = os.path.join(args.out, "context_{:06d}.png".format(result["frame"]))
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print("context plot saved to {}".format(os.path.relpath(path, ROOT)))
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
