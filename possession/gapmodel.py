"""Learned gap model (variant G, "v1e"): a logistic regression that says whether a ball-missing gap is dead ball,
trained on OTHER matches (Metrica Sports' open sample data), then applied to our match.

Pre-registered in docs/worklog.md ("2026-11-15 (learned gap model, v1e)"). Needs numpy only.

Features per gap (identical code for the sample games and for our match):
  log_len     ln(gap length in s)
  speed       median over the gap of the 3 s rolling median player speed; speed from position differences, <= 12 m/s
  edge_out    ball last seen within 2 m of a line, or outside the pitch
  restart_in  ball seen again at a restart spot (corner, centre spot, widened goal area, on or beyond a touchline)
  keeper      ball before and after in the same penalty area, seen again outside the goal area within 2 m of the
              defending team's deepest player (no tag condition: Smart Tagging does not exist for the sample games)
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np

import deadball
import gapscore

FPS = 25
FEATURES = ("log_len", "speed", "edge_out", "restart_in", "keeper")
MAX_SPEED = 12.0
L2 = 1.0
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "gap_model.json")


# ---------------------------------------------------------------- features (shared)

def speed_from_positions(pos):
    """(n, P) speed in m/s from consecutive positions; NaN where a slot is missing in either frame."""
    sp = np.full(pos.shape[:2], np.nan)
    sp[1:] = np.hypot(*(pos[1:] - pos[:-1]).transpose(2, 0, 1)) * FPS
    return np.minimum(sp, MAX_SPEED)


def keeper_team(before, after, pos_after, team, ends, pitch):
    """The defending team if the keeper pattern holds, else None."""
    if before is None or after is None:
        return None
    box_b = gapscore._in_box(before, pitch, gapscore.PEN_DEPTH, gapscore.PEN_HALF_WIDTH)
    box_a = gapscore._in_box(after, pitch, gapscore.PEN_DEPTH, gapscore.PEN_HALF_WIDTH)
    if box_b is None or box_b != box_a:
        return None
    if gapscore._in_box(after, pitch, gapscore.GOAL_AREA_DEPTH, gapscore.GOAL_AREA_HALF_WIDTH) is not None:
        return None
    owner = [t for t, end in ends.items() if end == box_a][0]
    k = gapscore.deepest_slot_xy(pos_after, team, owner, box_a, pitch)
    return owner if k is not None and np.hypot(*(k - after)) <= gapscore.KEEPER_M else None


def gap_table(pos, ball, team, halves, pitch):
    """One dict per gap inside each (first, last) half: s, e, length_s, the features, keeper_team."""
    roll = deadball.rolling_player_speed(speed_from_positions(pos))
    rows = []
    for s, e, first, last in gapscore.gaps_in(ball, halves):
        ends = gapscore.defended_end(pos, team, first, last)
        before = ball[s - 1] if s - 1 >= first else None
        after = ball[e + 1] if e + 1 <= last else None
        length_s = (e - s + 1) / FPS
        sp = roll[s:e + 1]
        kt = keeper_team(before, after, pos[e + 1] if e + 1 <= last else None, team, ends, pitch)
        rows.append(dict(s=s, e=e, length_s=length_s, log_len=np.log(length_s),
                         speed=float(np.nanmedian(sp)) if np.isfinite(sp).any() else np.nan,
                         edge_out=float(before is not None and gapscore.edge_out(before, pitch)),
                         restart_in=float(after is not None and gapscore.restart_spot(after, pitch)),
                         keeper=float(kt is not None), keeper_team=kt))
    return rows


def matrix(rows):
    return np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)


# ---------------------------------------------------------------- logistic regression (numpy, Newton)

def fit(X, y, l2=L2, iters=50):
    """Standardise, then L2-penalised logistic regression (intercept not penalised). Returns a model dict."""
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu, X)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    pen = np.r_[0.0, np.full(Z.shape[1] - 1, l2)]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        grad = Z.T @ (p - y) + pen * w
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return {"features": list(FEATURES), "mean": mu.tolist(), "sd": sd.tolist(), "intercept": float(w[0]),
            "coef": w[1:].tolist()}


def predict(model, X):
    mu, sd = np.array(model["mean"]), np.array(model["sd"])
    X = np.where(np.isnan(X), mu, X)
    z = model["intercept"] + ((X - mu) / sd) @ np.array(model["coef"])
    return 1 / (1 + np.exp(-z))


def save(model, path=MODEL_PATH):
    with open(path, "w") as handle:
        json.dump(model, handle, indent=2)


def load(path=MODEL_PATH):
    with open(path) as handle:
        return json.load(handle)


def apply(labels_d, rows, prob):
    """Variant G labels from variant D's labels and the gap probabilities."""
    out = labels_d.copy()
    for r, p in zip(rows, prob):
        if p >= 0.5:
            out[r["s"]:r["e"] + 1] = "D"
        elif r["keeper_team"] is not None:
            out[r["s"]:r["e"] + 1] = "HA"[r["keeper_team"]]
    return out


# ---------------------------------------------------------------- Metrica sample games (training data)

SAMPLE_DIR = os.path.join(os.path.dirname(HERE), "data", "external", "metrica_sample")
SAMPLE_PITCH = (105.0, 68.0)


def _restart(type_name, subtype):
    return type_name == "SET PIECE" or (subtype or "").startswith("GOAL KICK")


def _dead_from_events(events, n):
    """events: list of (type, subtype, start_frame, end_frame, period) with frames as row indices.
    Returns (dead mask, [(first, last)] per period)."""
    dead = np.zeros(n, dtype=bool)
    events = sorted(events, key=lambda x: (x[2], x[3]))
    halves = []
    for period in sorted({x[4] for x in events}):
        ev = [x for x in events if x[4] == period]
        halves.append((max(0, min(x[2] for x in ev)), min(n - 1, max(x[3] for x in ev))))
        for i, (tp, st, a, b, _) in enumerate(ev):
            if not _restart(tp, st) or i == 0:
                continue
            prev = [x for x in ev[:i] if x[0] not in ("CARD", "SET PIECE")]
            if prev:
                dead[max(0, max(x[3] for x in prev)):a] = True
    return dead, halves


def load_sample_csv(g):
    """Sample games 1 and 2 (CSV): pos (n, P, 2) m, team (P,), ball (n, 2) m, dead (n,), halves."""
    import csv
    pos, team = [], []
    ball = None
    for side, code in (("Home", 0), ("Away", 1)):
        path = os.path.join(SAMPLE_DIR, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(g, side))
        data = np.genfromtxt(path, delimiter=",", skip_header=3)
        xy = data[:, 3:]
        n_players = (xy.shape[1] - 2) // 2
        pos.append(xy[:, :2 * n_players].reshape(len(xy), n_players, 2))
        team += [code] * n_players
        ball = xy[:, -2:]
    pos = np.concatenate(pos, axis=1) * SAMPLE_PITCH
    ball = ball * SAMPLE_PITCH
    events = []
    with open(os.path.join(SAMPLE_DIR, "Sample_Game_{}_RawEventsData.csv".format(g))) as handle:
        for r in csv.DictReader(handle):
            a, b = int(r["Start Frame"]) - 1, int(r["End Frame"]) - 1
            events.append((r["Type"], r["Subtype"], a, max(a, b), int(r["Period"])))
    dead, halves = _dead_from_events(events, len(ball))
    return pos, np.array(team, dtype=np.int8), ball, dead, halves


def load_sample_epts():
    """Sample game 3 (EPTS txt + metadata XML + events JSON)."""
    root = ET.parse(os.path.join(SAMPLE_DIR, "Sample_Game_3_metadata.xml")).getroot()
    team_of = {p.get("id"): (0 if p.get("teamId") == "FIFATMA" else 1) for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    players = sorted(team_of)
    col = {p: i for i, p in enumerate(players)}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = [channel_player[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")
                 if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    with open(os.path.join(SAMPLE_DIR, "Sample_Game_3_tracking.txt")) as handle:
        lines = handle.read().splitlines()
    n = len(lines)
    pos = np.full((n, len(players), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for line in lines:
        frame, pl, bl = line.split(":")
        f = int(frame)
        order = next(o for a, b, o in specs if a <= f <= b)
        for pid, tok in zip(order, pl.split(";")):
            x, y = tok.split(",")
            if x != "NaN":
                pos[f - 1, col[pid]] = (float(x), float(y))
        bx, by = bl.split(",")[:2]
        if bx != "NaN":
            ball[f - 1] = (float(bx), float(by))
    with open(os.path.join(SAMPLE_DIR, "Sample_Game_3_events.json")) as handle:
        data = json.load(handle)["data"]
    events = []
    for e in data:
        st = e.get("subtypes")
        st = (st[0] if isinstance(st, list) and st else st) or {}
        a = int(e["start"]["frame"]) - 1
        b = int(e["end"]["frame"] or e["start"]["frame"]) - 1
        events.append((e["type"]["name"], st.get("name"), a, max(a, b), int(e["period"])))
    dead, halves = _dead_from_events(events, n)
    team = np.array([team_of[p] for p in players], dtype=np.int8)
    return pos * SAMPLE_PITCH, team, ball * SAMPLE_PITCH, dead, halves


def sample_gaps(g):
    """Gap rows with label y (1 = dead) and n_dead (dead frames) for sample game g."""
    pos, team, ball, dead, halves = load_sample_csv(g) if g in (1, 2) else load_sample_epts()
    rows = gap_table(pos, ball, team, halves, SAMPLE_PITCH)
    for r in rows:
        share = dead[r["s"]:r["e"] + 1].mean()
        r["y"] = float(share > 0.5)
        r["n_frames"] = r["e"] - r["s"] + 1
        r["n_dead"] = int(dead[r["s"]:r["e"] + 1].sum())
    return rows
