# Ball possession from tracking: controlled-ball evidence, hold and dead ball

Our submission to **More than a Hack 2027, Phase 1** (Metrica Sports, FC Barcelona, Mobile World Capital
Barcelona): for every video frame of FC Barcelona vs Athletic Club, label home possession (`H`), away possession
(`A`), dead ball (`D`) or not scored (`X`). The organisers' challenge description is kept unchanged in
[`ORGANISERS_README.md`](ORGANISERS_README.md), [`CHALLENGE.md`](CHALLENGE.md), [`RULES.md`](RULES.md),
[`SUBMISSION.md`](SUBMISSION.md) and [`EVALUATION.md`](EVALUATION.md).

**Technical document:** [`docs/technical_document.md`](docs/technical_document.md).

## Regenerate the submission file

The match data is not in this repository (it is not redistributable). Put the organisers' bundle in `data/` as
described in [`DATA.md`](DATA.md), then:

```bash
python3 -m venv .venv-submission
.venv-submission/bin/python -m pip install -r requirements-submission.txt     # numpy 2.4.1, pandas 2.3.3
.venv-submission/bin/python possession/submit.py --out <team_name>.csv
.venv-submission/bin/python evaluation/validate_submission.py <team_name>.csv
```

Expected: `rows 199319  H 53973  A 37061  D 67823  X 40462` and SHA-256
`8409a8920f34f0456d68aefdf47b9640804625173bc02d0ec866a2ca870870e5`. Deterministic: no learning, no randomness.
Details and other checks: [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

Tests (synthetic data, no match data needed):

```bash
python possession/test_submit.py
python possession/test_pipeline.py
```

## The method in one table

| Step | Rule |
|---|---|
| Who has the ball | A frame votes for a team only when exactly one team has a tracked player within 1.5 m of the ball and the ball is slower than 7 m/s (not in flight) |
| Smoothing | Centred majority vote over +/-60 frames inside each half; a tie keeps the previous team |
| Ball not tracked, under 10 s | The previous possession is held |
| Ball not tracked, 10 s or more | `D` |
| Stoppages | Players' rolling median speed under 1 m/s with the ball not moving, for 5 s or more: `D` |
| Ball out of play | Ball tracked beyond a touchline or goal line, until the restart kick: `D` |
| Outside the two halves | `X` |

The evidence rule was tested, pre-registered, on two other matches (Metrica's open sample games, event-derived
possession): spellF1 0.575 / 0.635 against 0.521 / 0.488 for the organisers' nearest-player rule. Gradient boosting
and a random forest trained on those matches tied with the rule and did not replace it. The full-match score of the
file is unknown.

## Repository map

| Path | What |
|---|---|
| `possession/submit.py` | **the submission generator** (method in its docstring) |
| `possession/tracking.py`, `deadball.py`, `outofplay.py`, `v1.py`, `v1b.py` | loader, stoppage detector, out-of-play detector, hold and long-gap rules used by it |
| `possession/baseline.py`, `report.py` | the organisers' baseline reproduced from scratch (0 of 199,319 frames differ) |
| `possession/evidence*.py`, `gbm_report.py`, `rf_report.py` | tests of the evidence rule and of learned models on Metrica's sample games |
| `possession/*_report.py`, `make_submission.py`, `gap*.py` | earlier candidates and their evaluations, kept so every number in the document can be re-run |
| `recovery/` | ball-recovery experiments (needs PyTorch; not in the submission path) |
| `explore/` | data exploration tools (local use; they read the data from `data/`) |
| `docs/results/` | every result table, with provenance (commit, code and data hashes) |
| `docs/worklog.md` | every decision, pre-registration and finding, newest first |
| `docs/paper/` | a short illustrated write-up (`paper.html`, figures from aggregates only) |
| `evaluation/`, `starter/` | the organisers' scorer, validator and starter code, unchanged |

## Licence

The organisers' files keep their licence ([`LICENSE`](LICENSE)); see [`CITATION.cff`](CITATION.cff).
