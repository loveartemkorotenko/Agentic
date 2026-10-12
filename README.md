# Air-raid alerts in Ukraine: onset and end-of-alert forecasting

Mini-project (KSE Stage 2): time-series analysis of air-raid alerts, per raion (118 raions + Kyiv City).
Two tasks, built with an AI assistant as the main engineering tool (see the AI log):

| | Task | Result |
|---|---|---|
| A | **Onset**: will an alert be active in a raion at any time in the next 6 h? | Small but statistically significant lift over strong baselines. Almost no skill at 1-3 h. |
| B | **End of alert**: an alert is active now, how much longer will it last? | Clear gains over baselines. This is the more useful and more learnable task. |

> **Research prototype, not safety advice.** Always follow official alerts. Never use a model estimate
> to decide when to leave a shelter.

## Data
Official alert history from
[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(`official_data_en.csv`, downloaded on first run). **The latest record is 2026-09-07**, so the test
periods end there.

Granularity changed over time: until mid-2025 alerts were published per oblast, since late 2025 almost only
per raion (3,000 oblast records in 2025-04 vs 48 in 2025-12). To keep one definition, an oblast-level alert
is expanded to all raions of that oblast and hromada alerts are mapped to their raion. (An early version that
filtered `level == "oblast"` silently dropped the entire recent period.) Training uses data from 2024-01-01.

## Leakage control
* Forecast origin `t` uses only grid steps (5 min) strictly before `t` and fully elapsed hours. The end time
  of a running alert is never a feature.
* Split `train | embargo | validation | embargo | test`; embargo = horizon (6 h for A, 24 h for B). Early
  stopping, thresholds, conformal calibration and tuning use validation only.
* `tests/test_leakage.py` erases every observation at/after `t` (grid length unchanged, cut exactly on an
  origin) and asserts identical features for all origins up to and including `t`;
  `tests/test_text_leakage.py` does the same for text features. Both were verified to **fail** when a
  one-step (alerts) / one-hour (text) look-ahead is injected.

## Task A: onset (6 h horizon)
LightGBM on current state, time since last alert, run length, rolling occupancy / episode counts (3-168 h),
same-hour-of-day profile (previous 7 days), oblast/national share in alert, calendar, raion id.
Baselines: persistence, same-hour-7d, 24 h mean occupancy, per-raion hour-of-day climatology.

Pooled numbers are inflated (9 frontline raions are almost always in alert), so the headline is the
**onset subset** (no alert at `t`), macro-averaged over raions with enough examples
(test 2026-07-09..2026-09-07, 81 raions):

| method | macro PR-AUC |
|---|---|
| **LightGBM** | **0.535** |
| occ_mean_24h | 0.521 |
| climatology | 0.474 |
| same_hour_7d | 0.462 |
| persistence | 0.456 (= base rate) |

Rolling-origin check (`python -m src.walk_forward`, 5 x 30-day windows): lift over the best baseline
+0.021, +0.033, +0.042, +0.016, +0.003 (mean +0.023, positive in 5/5 windows, **shrinking in the latest
window**). Week-block bootstrap over the pooled onset rows of all 23 test weeks: PR-AUC improvement
**+0.055, 95% CI [0.034, 0.072]**. (The Brier improvement, +0.083, is inflated because the occupancy
baseline is not a calibrated probability, so rank-based PR-AUC is the fair measure.)

Shorter horizons (1/2/3 h): lift over the best baseline is only +0.026 / +0.009 / +0.004, and Kyiv F1 at
1-2 h is about 0. Alert history alone does not contain the launch signal.

Ablations (all on the main split): dropping raion/oblast id features -> 0.529 (worse); data-driven
neighbour features (top-5 correlated raions, estimated on train only) -> 0.535 (no gain, optional
`--neighbors`).

## Task B: when does the alert end? (`python -m src.duration`)
At hourly origins where an alert is active, predict the remaining minutes (capped at 24 h: a few
frontline raions stay "in alert" for months; the cap avoids dominating the loss without selecting rows by
outcome) and P(ends within 60 / 120 min). LightGBM quantile regression (q10/q50/q90 on log1p minutes), binary
classifiers, and conformalised quantile regression (CQR) for the interval. Baselines use only the elapsed
alert time (bucketed), globally or per raion, estimated on the training period.

Test 2026-07-09..2026-09-07, 44,095 hourly origins:

| metric | LightGBM | best baseline (per-raion x elapsed) |
|---|---|---|
| MAE of median forecast | **248 min** | 297 min |
| P(end <= 60 min): PR-AUC | **0.727** | 0.645 |
| P(end <= 60 min): Brier | **0.144** | 0.169 |
| P(end <= 120 min): PR-AUC | **0.836** | 0.751 |

Paired week-block bootstrap (10 test weeks): MAE improvement **+48.7 min, 95% CI [37.6, 60.7]**; Brier
(60 min) improvement **+0.025, CI [0.022, 0.027]**.

Honest weak spot: the nominal 80% interval. Raw q10-q90 covers only 67% on test; after conformal
calibration on validation 75%, still below 80% because the regime shifts between validation and test (the
baseline intervals reach 78-80% but are narrower only in the point sense, i.e. they are not better).
Hyper-parameter tuning (random search on validation log-loss) and a LightGBM + logistic-regression ensemble
gave no gain on test (PR-AUC 0.725 / 0.729 vs 0.727), so the default parameters are kept.

## Optional: text data (public Telegram channels)
`python -m src.telegram <channel> --since 2026-06-01` scrapes the public web preview (~1 request/s,
publication timestamps only). Features per origin are counts of posts published before `t` that mention the
raion's oblast and/or a threat keyword (UAV / missile / KAB) plus national volume.
For Task A the channel `kpszsu` (22k posts) gave **no measurable gain** (macro PR-AUC with/without text, 6 h:
0.545 vs 0.538; 1-3 h within +-0.01). Likely reasons: reports appear almost simultaneously with the alert
(no lead time), crude keyword matching, only ~50 training days. See `reports/`.

## Limitations
* Single data source up to 2026-09-07; results depend on the period (see rolling-origin table).
* Task B intervals are not well calibrated under regime shift.
* Text: few channels, ~100 days; posts show current (possibly edited) text.
* Alert history cannot see the cause of an alert (launch events), so onset timing is the weak point.

## Run
```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m src.evaluate --horizon 6              # Task A, writes reports/results_h6.json
python -m src.walk_forward --horizon 6          # rolling origin + bootstrap
python -m src.duration                          # Task B (add --tune for random search)
```
