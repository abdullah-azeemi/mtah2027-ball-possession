# Reproducing the submission file

The possession file is produced by a deterministic pipeline (no randomness, no learning) from two files of
the provided data. Running it on the same data gives the same bytes.

## Environment

* Python 3.14.4 (developed and checked on macOS), `numpy` and `pandas` only:

```bash
python3 -m venv .venv-submission
.venv-submission/bin/python -m pip install -r requirements-submission.txt
```

* Checked in a clean virtual environment containing only these two packages (no matplotlib, no PyTorch).

## Data

Unpack the data bundle into `data/` as described in `DATA.md` (or set `MTAH_DATA_DIR`). The pipeline reads
only the tactical tracking:

* `data/tactical/BAR-ATH_tactical_atd.txt`   sha256 `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431`
* `data/tactical/BAR-ATH_tactical_atd.xml`   sha256 `327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a`

It does not read the video, the homography, the Smart Tagging or the broadcast files. The data is not in this
repository and must not be committed (`.gitignore` excludes `data/`).

## The command

```bash
.venv-submission/bin/python possession/submit.py --out <team_name>.csv
```

Expected output for the bundle above:

```
rows 199319  H 53973  A 37061  D 67823  X 40462
sha256 8409a8920f34f0456d68aefdf47b9640804625173bc02d0ec866a2ca870870e5
```

Then check the format with the organisers' validator:

```bash
.venv-submission/bin/python evaluation/validate_submission.py <team_name>.csv
```

(expected: valid, 334 possessions of mean 10.9 s counting each `X` as a gap).

Since 2026-11-15 `submit.py` also drops frame votes while the ball is in flight (faster than 7 m/s), tested on
Metrica's sample games (`docs/results/evidence_layer.md`). `--flight-speed 0` gives the original controlled-ball
file, SHA-256 `161bc1a36679f0e8d5e29581fe20862d5024206763371c15b03bb2d48d955d61`. Both hashes, and T_E's, were
reproduced with Python 3.14.4, numpy 2.4.0, pandas 2.3.3.

The submitted candidate is `possession/submit.py`: T_E's dead-ball rules unchanged, but a frame casts a
team vote only when **exactly one team has a tracked player within 1.5 m of the ball** (controlled
possession), with a +/-60 frame majority vote and ties to the previous decided possession. Method and
rationale are in the module docstring; evidence and caveats in `docs/results/submission.md`.

The previous candidate, T_E ("v1f"), is still in the repository for comparison:
`python possession/make_submission.py --out <team_name>.csv` gives
`rows 199319  H 52574  A 38460  D 67823  X 40462` and SHA-256
`fe38ac194ba507cdf8cc3ed4e2da26e907a2e4b65bf006dc7320f66131e5acb9` (493 possessions of mean 7.4 s).
The candidate before that, variant E alone, had SHA-256
`cda670e1a009a70009954a2701b7fd2cea08985b7c4b5403a4374174c22943b9`.

## Checks that were run

* Two runs give the same SHA-256, and the file equals the CSV written by `possession/submit_report.py`
  (checked 2026-11-15, in the development environment).
* The same SHA-256 with numpy 2.4.0, 2.4.1 and 2.4.2 (pandas 2.3.3) in clean virtual environments was
  checked for T_E; `submit.py` uses only the same numpy/pandas operations and reuses T_E's dead-ball code
  unchanged.
* `python possession/test_pipeline.py`: 10 checks of the building blocks on synthetic data, no match data.
* `python possession/test_submit.py`: 5 checks of the new evidence rule, vote ties, hold and the file
  contract on synthetic data, no match data.

## What the pipeline does

See `docs/technical_document.md` (approach, rules, limitations), the docstring of `possession/submit.py`
(the current submission) and `possession/make_submission.py` (the previous T_E candidate). The
experiments that were run and dropped are in `docs/worklog.md`; their code is in `explore/`, `recovery/`
and `possession/` (`recovery/` needs PyTorch and is not part of the submission path).
