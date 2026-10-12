# Air-raid alert forecasting for Ukrainian raions

Mini-project (KSE Stage 2): time-series analysis of air-raid alerts in Ukraine.

**Task.** At the start of every hour `t`, for every raion: will an air-raid alert be active
in that raion at any moment in `[t, t + 6h)`? Binary classification, 119 spatial units
(118 raions + Kyiv City), evaluated on the most recent 60 days of available data.

## Data
Official alert history from
[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(`official_data_en.csv`, downloaded automatically on first run).
**The latest record in the dataset is 2026-09-07**, so the test period is 2026-07-09 .. 2026-09-07.

Granularity changed over time: until mid-2025 alerts were published per oblast, since late
2025 almost exclusively per raion (e.g. 3,000 oblast records in 2025-04 vs 48 in 2025-12).
To keep one consistent definition, an oblast-level alert is expanded to all raions of that
oblast and hromada alerts are mapped to their raion. (An early version that filtered
`level == "oblast"` silently dropped the entire recent period.) Training uses data from 2024-01-01.

## Leakage control
* Forecast origin `t` uses only grid steps (5 min) strictly before `t` and fully elapsed hours.
  The end time of an ongoing alert is never used.
* Split is `train | embargo | validation | embargo | test` with embargo = horizon (6 h), so no
  training target window overlaps the evaluation period. Early stopping and F1 thresholds use
  validation only.
* `tests/test_leakage.py` recomputes all features after deleting every observation after a
  cut-off and asserts identical values for all origins up to the cut-off. It was verified to
  fail when a deliberate look-ahead is injected.

## Model
LightGBM on: current alert state, time since last alert, current alert duration, rolling
occupancy / episode counts over 3-168 h, same-hour-of-day profile over the previous 7 days,
share of raions in alert in the same oblast and nationally, hour-of-day, day-of-week, raion id.

Baselines: persistence, same hour over previous 7 days, 24 h mean occupancy, per-raion
hour-of-day climatology.

## Results (test: 2026-07-09 .. 2026-09-07, horizon 6 h)
Pooled over all raions, PR-AUC (positive rate 50.6%):
lightgbm 0.950, occ_mean_24h 0.929, climatology 0.913, same_hour_7d 0.885, persistence 0.753.

Pooled numbers are inflated because 9 frontline raions are in alert almost permanently and the
target is trivially 1. The honest headline is the **onset subset** (no alert active at `t`),
macro-averaged over 81 raions with enough examples of both classes:

| method | macro PR-AUC (onset) |
|---|---|
| **LightGBM** | **0.535** |
| occ_mean_24h | 0.521 |
| climatology | 0.474 |
| same_hour_7d | 0.462 |
| persistence | 0.456 (= base rate) |

LightGBM beats the best baseline in 67% of raions, but the margin over a simple 24 h occupancy
average is small: predicting *when a new wave starts* from alert history alone is hard.

Ablations (same split, `reports/`): dropping raion/oblast id features -> 0.529 (worse);
adding data-driven neighbour features (top-5 correlated raions, estimated on train only)
-> 0.535 (no gain, left optional via `--neighbors`).

## Robustness: rolling-origin validation (5 x 30-day windows, horizon 6 h)
`python -m src.walk_forward` retrains for each window (own train / 45-day validation / embargo).
Macro PR-AUC on onset rows, LightGBM minus best baseline: +0.021, +0.033, +0.042, +0.016, +0.003
(mean +0.023, positive in 5/5 windows). The lift is consistent but small and **shrinks in the most
recent window**, where LightGBM beats the best baseline in under half of raions. Absolute values
are not comparable across windows (different raions pass the min-positives filter).

Shorter horizons (1/2/3 h, single split): lift over the best baseline is +0.026 / +0.009 / +0.004,
i.e. almost none; for Kyiv F1 at 1-2 h is ~0. Alert history alone does not contain the launch signal.

## Optional: text data (public Telegram channel)
`python -m src.telegram kpszsu --since 2026-06-01` scrapes the public web preview of the channel
(22k posts, publication timestamps only, ~1 request/s). Features per origin `t` are counts of posts
published **before t** (fully elapsed hours) that mention the raion's oblast (Ukrainian place-name
stems) and/or a threat keyword (UAV / missile / KAB), plus national post volume and hours since
the last mention. `tests/test_text_leakage.py` checks that erasing posts at/after `t` does not
change any text feature (verified to fail with an injected 1-hour look-ahead).

Controlled comparison (same period 2026-06-01.., test = last 30 days, val 20 days, only ~50 days of
training because the text archive is short). Macro PR-AUC on onset rows, LightGBM without -> with text:

| horizon | no text | with text | best baseline |
|---|---|---|---|
| 1 h | 0.306 | 0.300 | 0.294 |
| 2 h | 0.403 | 0.402 | 0.410 |
| 3 h | 0.444 | 0.442 | 0.454 |
| 6 h | 0.538 | 0.545 | 0.568 |

**No measurable gain.** The hypothesis "text helps most at short horizons" was not supported.
Likely reasons: the channel reports threats roughly simultaneously with the alerts themselves, so
there is little lead time; keyword matching is crude (oblast-level, no raion localisation); and
with ~50 training days a gradient-boosted model cannot even beat a 24 h occupancy baseline.
The fair conclusion is "inconclusive, probably small", not "text works".

## Limitations
* Results vary with the period (see rolling-origin table).
* Text covers one channel and ~100 days; posts show current (possibly edited) text.
* Alert history cannot see the cause (launch events), so onset timing is the weak point.

## Run
```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m src.evaluate --horizon 6            # writes reports/results_h6.json
python -m src.evaluate --horizon 6 --neighbors --tag _neighbors
```
