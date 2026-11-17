"""Load the tactical tracking for ALL video frames into arrays, plus the team of every slot.

Everything here is at the full 25 Hz and indexed by video frame (= submission row). Positions are
returned in metres (normalised coordinates times the pitch size 105.3 x 68.0), as float64: the
reference baseline computes its distances in float64 and we want to reproduce its labels exactly,
including near-ties.

    pos   (199319, 46, 2)  slot positions in metres, NaN where the slot has no position
    ball  (199319, 2)      ball position in metres, NaN where the ball is not tracked
    team  (46,)            0 = home (Barcelona), 1 = away (Athletic), from the tracking metadata

Line i of the raw file holds frame_id i + 1 (checked on every line), so row i is video frame i.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH = GAMES[GAME]["pitch"]
N_SLOTS = 46
HOME, AWAY = 0, 1


def team_of_slots(meta_path):
    """Array over the 46 slots: 0 for the home team, 1 for the away team.

    The metadata lists two teams and assigns every Player (track slot) to one of them. The first
    team listed is the home side (FIFA/EPTS convention; for this match Barcelona, which DATA.md
    states independently). A slot is a tracklet, not a person.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if len(slots) != N_SLOTS:
        raise ValueError("expected {} slots, found {}".format(N_SLOTS, len(slots)))
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def load_tracking(data_dir=None, with_speed=False):
    """(pos, ball, team) for every video frame; see the module docstring.

    With ``with_speed=True`` a fourth array is returned: ``speed`` (199319, 46), the speed in m/s that
    the tracking file reports for each slot (NaN where the slot has no position or no speed value).
    """
    paths = asset_paths(data_dir=data_dir)
    pos = np.full((N_FRAMES, N_SLOTS, 2), np.nan)
    speed = np.full((N_FRAMES, N_SLOTS), np.nan) if with_speed else None
    ball = np.full((N_FRAMES, 2), np.nan)
    seen = np.zeros(N_FRAMES, dtype=bool)
    with open(paths["atd_raw"], "r") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            seen[frame] = True
            b = parts[2].split(",")
            if not b[0].startswith("N"):
                ball[frame] = (float(b[0]), float(b[1]))
            for slot, token in enumerate(parts[1].split(";")):
                if token[0] != "N":
                    x, y, s = token.split(",")
                    pos[frame, slot] = (float(x), float(y))
                    if with_speed:
                        speed[frame, slot] = float(s)
    if not seen.all():
        raise ValueError("{} frames missing from the tracking file".format(int((~seen).sum())))
    scale = np.array(PITCH)
    result = (pos * scale, ball * scale, team_of_slots(paths["atd_meta"]))
    return result + (speed,) if with_speed else result
