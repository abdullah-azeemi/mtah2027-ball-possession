"""Ball recovery model: a Set Transformer over the players, then a two-stage (hierarchical) Bi-LSTM.

This follows the structure of Kim et al., "Ball Trajectory Inference from Multi-Agent Sports
Contexts Using Set Transformer and Hierarchical Bi-LSTM" (KDD 2023, arXiv:2306.08206), adapted to
what this competition's data allows. What is kept, what is changed, and why:

Kept
  * Players at each time step are a SET: a Set Transformer (self-attention with no positional
    order, plus pooling by attention) encodes them, so the model does not care which track slot a
    player has. That matters here: slots swap identity (worklog finding 6).
  * Hierarchy: stage 1 predicts WHO has the ball (a distribution over the players at each step),
    stage 2 predicts WHERE the ball is, given that possession estimate. Both are Bi-LSTMs over time.
  * Per-player input features: position, velocity, speed, acceleration (the paper found each one
    helps), plus the team.
  * Hidden ball stretches are filled in from the players and from the ball where it is tracked.

Changed
  * The paper trains the possession stage on possession labels built from human-annotated event
    data. We may not use human labels or event data for this match (RULES.md). Our possession
    targets are produced by an algorithm: the player nearest to the TRACKED ball, if within 3 m,
    else "nobody". They are pseudo-labels, not truth.
  * The paper feeds players as 22 fixed, ordered slots; we feed up to 46 track slots with a mask
    for the missing ones, plus one always-present "nobody has it" token.
  * No "reality loss" (the paper's penalty on turning when no player is near): the formula as
    printed there has an unresolved sign (see worklog), so we do not copy it blindly.
  * Smaller (about 1M parameters instead of 8.7M) because the data is one match, not a dozen.

Input / output
  pos       (B, T, 46, 2)  player positions in metres, NaN = slot not tracked
  ball_in   (B, T, 2)      ball in metres where visible to the model (NaN allowed)
  hidden    (B, T)         True where the ball is hidden from the model
  team      (46,)          0 / 1 team of each slot
  returns   ball_hat (B, T, 2) in metres, possession_logits (B, T, 47) (46 slots + "nobody")
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import HZ, N_SLOTS, PITCH_L, PITCH_W

CENTER = (PITCH_L / 2.0, PITCH_W / 2.0)
SCALE = 50.0            # metres; coordinates are fed to the network as (m - centre) / 50
N_FEATURES = 8
NOBODY = N_SLOTS        # index of the "nobody has the ball" token


def player_features(pos):
    """(B,T,S,2) metres with NaN -> features (B,T,S,8) and the presence mask (B,T,S).

    Features: x, y (centred, scaled), velocity (2), speed, acceleration (2), present flag. Velocity
    and acceleration come from finite differences between consecutive steps and are 0 when the
    player is not tracked in both steps.
    """
    present = torch.isfinite(pos[..., 0])
    p = torch.nan_to_num(pos, nan=0.0)
    dt = 1.0 / HZ
    prev, prev_present = torch.roll(p, 1, dims=1), torch.roll(present, 1, dims=1)
    ok = present & prev_present
    ok[:, 0] = False
    v = torch.where(ok[..., None], (p - prev) / dt, torch.zeros_like(p))
    prev_ok = torch.roll(ok, 1, dims=1) & ok
    prev_ok[:, 0] = False
    a = torch.where(prev_ok[..., None], (v - torch.roll(v, 1, dims=1)) / dt, torch.zeros_like(p))
    centre = torch.tensor(CENTER, device=pos.device, dtype=pos.dtype)
    xy = torch.where(present[..., None], (p - centre) / SCALE, torch.zeros_like(p))
    speed = v.norm(dim=-1, keepdim=True)
    feats = torch.cat([xy, v / 10.0, speed / 10.0, a.clamp(-20, 20) / 10.0,
                       present[..., None].to(pos.dtype)], dim=-1)
    return feats, present


class SAB(nn.Module):
    """Self-attention block (pre-norm) over a set; padded elements are ignored as keys."""

    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, pad):
        h = self.ln1(x)
        x = x + self.attention(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class PlayerSetEncoder(nn.Module):
    """Set Transformer encoder: per-player tokens and a pooled context vector for every time step."""

    def __init__(self, d=64, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(N_FEATURES, d)
        self.team = nn.Embedding(2, d)
        self.null = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([SAB(d, heads) for _ in range(layers)])
        self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)       # pooling by attention (PMA)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, feats, present, team):
        b, t, s, _ = feats.shape
        x = self.inp(feats) + self.team(team)[None, None]
        null = self.null.expand(b, t, 1, -1)
        x = torch.cat([x, null], dim=2).reshape(b * t, s + 1, -1)
        pad = torch.cat([~present, torch.zeros_like(present[..., :1])], dim=2).reshape(b * t, s + 1)
        for block in self.blocks:
            x = block(x, pad)
        pooled = self.pool(self.seed.expand(b * t, -1, -1), x, x, key_padding_mask=pad,
                           need_weights=False)[0]
        return x.reshape(b, t, s + 1, -1), pad.reshape(b, t, s + 1), self.ln(pooled).reshape(b, t, -1)


def interpolate_ball(ball_in, hidden, pos, two_sided=True):
    """The model-free guess for every step, (B,T,2) metres, computed from what the model may see.

    Visible steps keep their value. A hidden step between two visible ones is linearly interpolated
    (``two_sided``) or held at the last visible value (past only). Before the first / after the last
    visible step the nearest visible value is held. If the whole window is hidden the guess is the
    centroid of the tracked players at that step (pitch centre if there are none).
    """
    b, t, _ = ball_in.shape
    device, dtype = ball_in.device, ball_in.dtype
    visible = (~hidden) & torch.isfinite(ball_in[..., 0])
    idx = torch.arange(t, device=device).expand(b, t)
    far = t + 10
    last = torch.cummax(torch.where(visible, idx, torch.full_like(idx, -1)), dim=1).values
    nxt = torch.flip(torch.cummin(
        torch.flip(torch.where(visible, idx, torch.full_like(idx, far)), [1]), dim=1).values, [1])
    has_last, has_next = last >= 0, nxt < far

    values = torch.nan_to_num(ball_in, nan=0.0)
    take = lambda i: torch.gather(values, 1, i.clamp(0, t - 1)[..., None].expand(-1, -1, 2))  # noqa: E731
    p_last, p_next = take(last), take(nxt)

    present = torch.isfinite(pos[..., 0])
    count = present.sum(dim=2, keepdim=True)
    centre = torch.tensor(CENTER, device=device, dtype=dtype)
    centroid = torch.where(count > 0, torch.nan_to_num(pos, nan=0.0).sum(dim=2) / count.clamp(min=1),
                           centre.expand(b, t, 2))

    span = (nxt - last).clamp(min=1).to(dtype)
    weight = ((idx - last).to(dtype) / span)[..., None]
    between = p_last * (1 - weight) + p_next * weight
    if two_sided:
        guess = torch.where((has_last & has_next)[..., None], between,
                            torch.where(has_last[..., None], p_last,
                                        torch.where(has_next[..., None], p_next, centroid)))
    else:
        guess = torch.where(has_last[..., None], p_last, centroid)
    return guess


class BallRecoverer(nn.Module):
    """residual=False: the network outputs the ball position directly.
    residual=True:  it outputs a correction added to the model-free interpolation of the visible
                    ball, and also receives that interpolation as an input. It can then only add
                    what the players tell it; it starts out equal to the interpolation."""

    def __init__(self, d=64, hidden=128, layers=2, dropout=0.2, bidirectional=True, residual=False):
        super().__init__()
        self.d, self.residual, self.bidirectional = d, residual, bidirectional
        extra = 2 if residual else 0
        self.encoder = PlayerSetEncoder(d)
        mult = 2 if bidirectional else 1
        # Stage 1: who has the ball?   input = pooled players + the ball as the model may see it
        self.rnn1 = nn.LSTM(d + 3 + extra, hidden, layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        self.query = nn.Linear(hidden * mult, d)
        self.key = nn.Linear(d, d)
        # Stage 2: where is it?        input = players + soft possessor position + possession info
        self.rnn2 = nn.LSTM(d + 6 + extra, hidden, layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        self.head = nn.Linear(hidden * mult, 2)
        if residual:
            nn.init.normal_(self.head.weight, std=0.01)     # start close to the interpolation
            nn.init.zeros_(self.head.bias)

    @property
    def label(self):
        return ("Set Transformer + " + ("Bi-LSTM" if self.bidirectional else "LSTM (past only)")
                + (" + residual on interpolation" if self.residual else " (direct)"))

    def forward(self, pos, ball_in, hidden, team):
        feats, present = player_features(pos)
        tokens, pad, context = self.encoder(feats, present, team)

        centre = torch.tensor(CENTER, device=pos.device, dtype=pos.dtype)
        visible = (~hidden).to(pos.dtype)[..., None]
        ball = torch.nan_to_num((ball_in - centre) / SCALE, nan=0.0) * visible
        if self.residual:
            guess = interpolate_ball(ball_in, hidden, pos, two_sided=self.bidirectional)
            guess_n = (guess - centre) / SCALE
            extra = [guess_n]
        else:
            extra = []

        u, _ = self.rnn1(torch.cat([context, ball, visible] + extra, dim=-1))
        q = self.query(u)                                                  # (B,T,d)
        k = self.key(tokens)                                               # (B,T,S+1,d)
        logits = (q[:, :, None, :] * k).sum(-1) / math.sqrt(self.d)
        logits = logits.masked_fill(pad, -1e4)
        g = F.softmax(logits, dim=-1)                                      # who has it

        slot_xy = torch.nan_to_num((pos - centre) / SCALE, nan=0.0)
        soft_possessor = (g[..., :N_SLOTS, None] * slot_xy).sum(dim=2)     # (B,T,2)

        v, _ = self.rnn2(torch.cat([context, soft_possessor, g[..., NOBODY:], ball, visible] + extra,
                                   dim=-1))
        if self.residual:
            ball_hat = guess + self.head(v) * SCALE      # correction in metres on top of the guess
        else:
            ball_hat = self.head(v) * SCALE + centre
        return ball_hat, logits


def possession_pseudo_labels(pos, ball_true, radius=3.0):
    """Algorithmic possession targets (B,T): nearest tracked player within `radius` m of the
    tracked ball, NOBODY otherwise, -100 (ignored) where the ball is not tracked."""
    dist = (pos - ball_true[:, :, None, :]).norm(dim=-1)
    dist = torch.nan_to_num(dist, nan=float("inf"))
    nearest = dist.argmin(dim=-1)
    close = dist.gather(-1, nearest[..., None])[..., 0] < radius
    labels = torch.where(close, nearest, torch.full_like(nearest, NOBODY))
    return torch.where(torch.isfinite(ball_true[..., 0]), labels, torch.full_like(labels, -100))
