"""List 10 long ball-missing gaps for a MANUAL look at the tactical video (is the ball visible?).

Selection rule, fixed before running: take the ball-missing gaps of at least 10 s that lie fully inside
the two graded halves, sort by start frame, and take 10 evenly spaced ones (index round(linspace(0, n-1, 10))).
Only frame numbers and video times are written. No tracking values, no images.
Usage: python possession/gap_check_list.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracking import load_tracking  # noqa: E402

FPS = 25
HALVES = [(11185, 90651), (110808, 190197)]

_, ball, _ = load_tracking()
miss = np.isnan(ball[:, 0])
gaps = []
for a, b in HALVES:
    m = miss[a:b + 1].astype(int)
    d = np.diff(np.r_[0, m, 0])
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
        if (e - s + 1) >= 10 * FPS:
            gaps.append((a + s, a + e))
gaps.sort()
idx = np.unique(np.round(np.linspace(0, len(gaps) - 1, 10)).astype(int))
mmss = lambda f: "%d:%02d" % divmod(int(f / FPS), 60)
lines = ["# Gap check list: is the ball visible in the video?", "",
         "%d ball-missing gaps of at least 10 s lie inside the graded halves; 10 evenly spaced ones by start frame:" % len(gaps), "",
         "| # | start frame | end frame | length (s) | video time (start to end) | ball visible? (fill in) |", "|---|---|---|---|---|---|"]
for k, i in enumerate(idx, 1):
    s, e = gaps[i]
    lines.append("| %d | %d | %d | %.1f | %s to %s | |" % (k, s, e, (e - s + 1) / FPS, mmss(s), mmss(e)))
lines += ["", "Look at the middle and both ends of each gap. Write: yes (clear), partly, no (hidden or out of play).",
          "Video time is frame / 25 s from the start of the tactical video."]
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "results", "gap_check_list.md")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
