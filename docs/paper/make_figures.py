"""Figures for docs/paper. Aggregates and illustrations only: no frame-level data export, no video frames
(RULES.md: do not publish derived data that would reconstruct the data). Frame illustration uses MADE-UP positions.
Run: python docs/paper/make_figures.py   (needs the data in ./data, numpy, matplotlib)"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "possession")); sys.path.insert(0, ROOT)
import tracking, baseline, v1, v1b  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

OUT = os.path.join(HERE, "figures")
BLUE, RED, GREY, GREEN = "#1f5fbf", "#c0392b", "#8a8f98", "#2e8b57"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

def pitch(ax):
    ax.add_patch(Rectangle((0, 0), 105, 68, fill=False, ec="#444", lw=1.2))
    ax.plot([52.5, 52.5], [0, 68], c="#444", lw=1); ax.add_patch(Circle((52.5, 34), 9.15, fill=False, ec="#444", lw=1))
    for x in (0, 88.5): ax.add_patch(Rectangle((x, 13.8), 16.5, 40.3, fill=False, ec="#444", lw=1))
    ax.set_xlim(-3, 108); ax.set_ylim(-3, 71); ax.set_aspect("equal"); ax.axis("off")

# 1. illustrative frame (made-up positions)
rng = np.random.default_rng(7)
home = np.c_[rng.uniform(25, 75, 10), rng.uniform(8, 60, 10)]; away = np.c_[rng.uniform(25, 75, 10), rng.uniform(8, 60, 10)]
ball = np.array([58.0, 31.0])
allp = np.vstack([home, away]); d = np.hypot(*(allp - ball).T); k = int(d.argmin())
fig, ax = plt.subplots(figsize=(7, 4.6)); pitch(ax)
ax.scatter(*home.T, c=BLUE, s=70, label="home slots", zorder=3); ax.scatter(*away.T, c=RED, s=70, label="away slots", zorder=3)
ax.scatter(*ball, c="black", s=60, marker="o", zorder=4, label="ball (tracked)")
ax.plot([ball[0], allp[k, 0]], [ball[1], allp[k, 1]], c="black", lw=1.5, ls="--", zorder=2)
ax.annotate("nearest slot: %.1f m\n-> its team has the ball" % d[k], allp[k], (allp[k, 0] + 6, allp[k, 1] + 14), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.set_title("Illustration with made-up positions: the per-frame evidence", fontsize=10)
ax.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04)); fig.savefig(os.path.join(OUT, "fig1_frame_rule.png"), bbox_inches="tight"); plt.close(fig)

pos, ball_a, team, speed = tracking.load_tracking(with_speed=True)
n = len(ball_a); graded = np.zeros(n, bool); blocks = list(half_video_blocks(GAME))
for _, a, b in blocks: graded[a:b + 1] = True
miss = np.isnan(ball_a[:, 0]); ntrk = np.isfinite(pos[:, :, 0]).sum(1)

# 2. ball missing per minute
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.2), sharey=True)
for ax, (nm, a, b) in zip(axs, blocks):
    m = [miss[s:s + 1500].mean() * 100 for s in range(a, b - 1499, 1500)]
    ax.bar(range(len(m)), m, color=GREY); ax.set_ylabel("ball missing (%)"); ax.set_title("Half %s" % nm if len(str(nm)) < 3 else str(nm), fontsize=9, loc="left")
axs[1].set_xlabel("minute of the half"); fig.suptitle("Ball missing per minute (scored frames; 40.7% overall)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_missing_per_minute.png"), bbox_inches="tight"); plt.close(fig)

# 3. gap length histogram
lens = []
for _, a, b in blocks:
    m = miss[a:b + 1].astype(int); dd = np.diff(np.r_[0, m, 0])
    lens += list((np.where(dd == -1)[0] - np.where(dd == 1)[0]) / 25)
lens = np.array(lens)
fig, ax = plt.subplots(figsize=(6.2, 3.2)); ax.hist(np.clip(lens, 0, 100), bins=[0, 1, 2, 5, 10, 20, 30, 60, 100], color=GREY, edgecolor="white")
ax.axvline(10, c=RED, ls="--"); ax.text(10.5, ax.get_ylim()[1] * 0.85, "10 s rule", color=RED)
ax.set_xlabel("gap length (s), values over 100 s clipped"); ax.set_ylabel("number of gaps"); ax.set_title("%d ball gaps, median %.1f s" % (len(lens), np.median(lens)), fontsize=10)
fig.savefig(os.path.join(OUT, "fig3_gap_lengths.png"), bbox_inches="tight"); plt.close(fig)

# 4. missing vs slots tracked
bins = [("<=16", ntrk <= 16)] + [(str(k), ntrk == k) for k in range(17, 23)] + [(">=23", ntrk >= 23)]
vals = [miss[graded & m].mean() * 100 for _, m in bins]
fig, ax = plt.subplots(figsize=(6.2, 3.2)); ax.bar([b for b, _ in bins], vals, color=BLUE)
ax.set_xlabel("slots tracked in the frame"); ax.set_ylabel("ball missing (%)"); ax.set_title("Missingness is not random: fewer tracked slots, more missing ball", fontsize=10)
fig.savefig(os.path.join(OUT, "fig4_missing_vs_slots.png"), bbox_inches="tight"); plt.close(fig)

# 5. recovery error vs gap length (numbers from docs/results/ball_recovery_*.md; mean error, linear interpolation)
fig, ax = plt.subplots(figsize=(6.2, 3.2)); ax.plot([1, 2, 5, 10], [0.8, 2.2, 6.3, 12], "o-", c=GREEN, label="linear interpolation")
ax.axvline(10, c=GREY, ls=":"); ax.text(10.2, 2, "most real gaps\nare longer", fontsize=8, color=GREY)
ax.set_xlabel("hidden stretch (s)"); ax.set_ylabel("mean ball error (m)"); ax.set_title("Reconstructing a hidden ball (neural model: not better)", fontsize=10); ax.legend(frameon=False)
fig.savefig(os.path.join(OUT, "fig5_recovery_error.png"), bbox_inches="tight"); plt.close(fig)

# 6. D share per minute: reference vs submitted
codes = baseline.nearest_team(pos, ball_a, team); vote = v1b.votes(codes, v1b.WINDOW); dead = v1.dead_flags(speed, ball_a, graded)
A = v1b.build(vote, dead, False, False)
import make_submission; E = make_submission.make_labels()   # submitted T_E (v1f)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.2), sharey=True)
for ax, (nm, a, b) in zip(axs, blocks):
    xs = range(0, (b - a + 1) // 1500)
    ax.plot(xs, [(A[a + i * 1500:a + (i + 1) * 1500] == "D").mean() * 100 for i in xs], c=GREY, label="reference (smooth50)")
    ax.plot(xs, [(E[a + i * 1500:a + (i + 1) * 1500] == "D").mean() * 100 for i in xs], c=RED, label="submitted (T_E)")
    ax.set_ylabel("labelled D (%)")
axs[0].legend(frameon=False, ncol=2, fontsize=8); axs[1].set_xlabel("minute of the half"); fig.suptitle("Share of frames labelled D per minute", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_dead_share.png"), bbox_inches="tight"); plt.close(fig)

# 7. public window scores (from the official scorer, 1000 bootstrap draws)
names = ["smooth50", "smooth125", "ours T_E", "ours (submitted)"]; sp = [0.3256, 0.4242, 0.3905, 0.5455]
lo = [0.2140, None, 0.2571, 0.3879]; hi = [0.4465, None, 0.5238, 0.7030]
fig, ax = plt.subplots(figsize=(6.2, 3.2)); cols = [GREY, GREY, GREY, RED]; ax.bar(names, sp, color=cols)
for i, (l, h) in enumerate(zip(lo, hi)):
    if l is not None: ax.plot([i, i], [l, h], c="black", lw=1.5)
ax.axhline(0.2433, c=BLUE, ls="--", label="0.2433 = published floor on the FULL match (not comparable)"); ax.legend(loc="upper left", frameon=False, fontsize=7.5); ax.set_ylim(0, 0.75)
ax.set_ylabel("spellF1, public 5 min"); ax.set_title("Public 5 min only: 17 possessions, intervals overlap", fontsize=10)
fig.savefig(os.path.join(OUT, "fig7_public_scores.png"), bbox_inches="tight"); plt.close(fig)

# 8. SHAP of a random forest WITHOUT the rule's label, Metrica sample games (docs/results/rf_shap.md, mean |SHAP|
# over 2,000 held-out live frames per game; averaged over the two test games). Public sample data, aggregates only.
shap_rows = [("nearest away player", 0.0682, 0.0772), ("nearest away, 1 s mean", 0.0638, 0.0565),
             ("nearest home, 1 s mean", 0.0581, 0.0623), ("home within 3 m, 1 s mean", 0.0650, 0.0467),
             ("nearest home player", 0.0482, 0.0511), ("away within 3 m, 1 s mean", 0.0502, 0.0449),
             ("home within 1.5 m", 0.0473, 0.0476), ("away within 1.5 m", 0.0409, 0.0518),
             ("ball speed", 0.0113, 0.0145), ("seconds since ball seen", 0.0035, 0.0050)]
shap_rows = sorted(shap_rows, key=lambda r: r[1] + r[2])
labels = [r[0] for r in shap_rows]; vals = [(r[1] + r[2]) / 2 for r in shap_rows]
cols = [RED if ("1.5 m" in l or "nearest" in l or "speed" in l) else GREY for l in labels]
fig, ax = plt.subplots(figsize=(6.2, 3.0)); ax.barh(labels, vals, color=cols)
ax.set_xlabel("mean |SHAP| for P(home), held-out sample game"); ax.tick_params(axis="y", labelsize=8)
ax.set_title("What a learned model uses (red: the signals our rule uses)", fontsize=10)
fig.savefig(os.path.join(OUT, "fig8_shap.png"), bbox_inches="tight"); plt.close(fig)
print("figures written to", OUT)
