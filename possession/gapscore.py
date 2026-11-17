"""Gap scorecard (variant F, "v1d"): decide, gap by gap, whether a ball-missing stretch is dead ball, live play
held by a goalkeeper, or live play to be held as before. No learning, no labels.

Rule pre-registered in docs/worklog.md ("2026-11-15 (gap scorecard, v1d)"); every constant below is copied from
there and must not be changed without a new dated pre-registration.

A gap is a maximal run of frames with the ball untracked inside one graded half: first frame s, last frame e.
Features (pitch 105.3 x 68.0 m, origin at a corner):

  long        length >= 10 s
  slow        median over the gap of the 3 s rolling median player speed < 1.0 m/s
  edge_out    ball at s-1 within 2 m of a touchline or goal line, or outside the pitch
  restart_in  ball at e+1 at a restart spot (corner, centre spot, widened goal area, on or beyond a touchline)
  tag         a Smart Tagging SET PIECES tag starts in [s - 3 s, e + 3 s]
  keeper      no tag; ball at s-1 and e+1 inside the same penalty area, owned by team T; at e+1 the ball is outside
              the goal area and within 2 m of T's deepest tracked slot

  points = 2 long + 2 slow + 1 edge_out + 1 restart_in + 2 tag - 3 keeper
  points >= 2 -> D;  keeper -> team T on the whole gap;  otherwise the gap keeps variant D's labels.
"""

import numpy as np

import deadball

FPS = 25
LONG_S = 10
SLOW_MPS = 1.0
EDGE_M = 2.0
CORNER_M = 3.0
CENTRE_M = 3.0
GOAL_AREA_MARGIN_M = 1.0
TOUCHLINE_M = 1.0
TAG_MARGIN_S = 3
KEEPER_M = 2.0
POINTS = {"long": 2, "slow": 2, "edge_out": 1, "restart_in": 1, "tag": 2, "keeper": -3}
THRESHOLD = 2

PEN_DEPTH, PEN_HALF_WIDTH = 16.5, 20.16
GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH = 5.5, 9.16


def gaps_in(ball, halves):
    """[(s, e, first, last)] of ball-missing runs inside each half (first, last = the half's bounds)."""
    miss = np.isnan(ball[:, 0])
    out = []
    for first, last in halves:
        m = miss[first:last + 1].astype(int)
        d = np.diff(np.r_[0, m, 0])
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
            out.append((first + s, first + e, first, last))
    return out


def _in_box(p, pitch, depth, half_width, margin=0.0):
    """Which end's box (0 = x near 0, 1 = x near length) contains point p, or None."""
    length, width = pitch
    if abs(p[1] - width / 2) > half_width + margin:
        return None
    if -margin <= p[0] <= depth + margin:
        return 0
    if length - depth - margin <= p[0] <= length + margin:
        return 1
    return None


def edge_out(p, pitch):
    length, width = pitch
    return bool(p[0] <= EDGE_M or p[0] >= length - EDGE_M or p[1] <= EDGE_M or p[1] >= width - EDGE_M)


def restart_spot(p, pitch):
    length, width = pitch
    corners = np.array([(0, 0), (0, width), (length, 0), (length, width)], dtype=float)
    if (np.hypot(*(corners - p).T) <= CORNER_M).any():
        return True
    if np.hypot(p[0] - length / 2, p[1] - width / 2) <= CENTRE_M:
        return True
    if _in_box(p, pitch, GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH, GOAL_AREA_MARGIN_M) is not None:
        return True
    return bool(p[1] <= TOUCHLINE_M or p[1] >= width - TOUCHLINE_M)


def defended_end(pos, team, first, last):
    """{team: end} for one half: the team whose slots have the lower mean x defends end 0 (x near 0)."""
    x = pos[first:last + 1, :, 0]
    mean_x = [np.nanmean(x[:, team == t]) for t in (0, 1)]
    low = int(np.argmin(mean_x))
    return {low: 0, 1 - low: 1}


def deepest_slot_xy(pos_frame, team, t, end, pitch):
    """Position of team t's tracked slot nearest its own goal line, or None."""
    p = pos_frame[team == t]
    p = p[np.isfinite(p[:, 0])]
    if not len(p):
        return None
    dist = p[:, 0] if end == 0 else pitch[0] - p[:, 0]
    return p[int(np.argmin(dist))]


def gap_features(pos, ball, team, roll_speed, tag_starts, halves, pitch):
    """One dict per gap: frames, length, features, points, keeper team (or None) and decision."""
    rows = []
    ends = {(a, b): defended_end(pos, team, a, b) for a, b in halves}
    for s, e, first, last in gaps_in(ball, halves):
        length_s = (e - s + 1) / FPS
        before = ball[s - 1] if s - 1 >= first else None
        after = ball[e + 1] if e + 1 <= last else None
        f = {"long": length_s >= LONG_S}
        sp = roll_speed[s:e + 1]
        f["slow"] = bool(np.isfinite(sp).any() and np.nanmedian(sp) < SLOW_MPS)
        f["edge_out"] = before is not None and edge_out(before, pitch)
        f["restart_in"] = after is not None and restart_spot(after, pitch)
        f["tag"] = bool(((tag_starts >= s - TAG_MARGIN_S * FPS) & (tag_starts <= e + TAG_MARGIN_S * FPS)).any())
        keeper_team = None
        if not f["tag"] and before is not None and after is not None:
            box_b = _in_box(before, pitch, PEN_DEPTH, PEN_HALF_WIDTH)
            box_a = _in_box(after, pitch, PEN_DEPTH, PEN_HALF_WIDTH)
            in_goal_area = _in_box(after, pitch, GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH) is not None
            if box_b is not None and box_b == box_a and not in_goal_area:
                owner = [t for t, end in ends[(first, last)].items() if end == box_a][0]
                k = deepest_slot_xy(pos[e + 1], team, owner, box_a, pitch)
                if k is not None and np.hypot(*(k - after)) <= KEEPER_M:
                    keeper_team = owner
        f["keeper"] = keeper_team is not None
        points = sum(POINTS[k] for k, v in f.items() if v)
        decision = "keeper" if f["keeper"] else ("D" if points >= THRESHOLD else "keep")
        rows.append(dict(s=s, e=e, length_s=length_s, points=points, keeper_team=keeper_team, decision=decision, **f))
    return rows


def apply(labels_d, rows):
    """Variant F labels from variant D's labels (vote + hold + detector) and the gap decisions."""
    out = labels_d.copy()
    for r in rows:
        if r["decision"] == "D":
            out[r["s"]:r["e"] + 1] = "D"
        elif r["decision"] == "keeper":
            out[r["s"]:r["e"] + 1] = "HA"[r["keeper_team"]]
    return out


def tag_start_frames(restart_tags):
    return np.array([start for _, start in restart_tags], dtype=np.int64)


def rolling_speed(speed):
    return deadball.rolling_player_speed(speed)
