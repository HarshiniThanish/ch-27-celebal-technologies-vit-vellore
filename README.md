# Hub-Level Daily Order Volume Forecasting — Approach Summary

## 1. Problem framing

The task is a **42-day-ahead demand forecast across 1,115 fulfillment hubs**
(46,830 hub-day predictions), evaluated on **RMSLE**. Training data covers
2013-01-01 to 2015-06-19; the test window is 2015-06-20 to 2015-07-31.
Because future order volumes are unknown at prediction time, the problem is
treated as **1,115 parallel time series requiring recursive multi-step
forecasting**, not a single-step tabular regression.

## 2. Data preparation

- Merged `orders_train` / `orders_test` with `hub_metadata` on `HubID`.
- No duplicate `HubID`+`Date` rows; no missing values in the daily order
  data. Missingness is confined to hub metadata (e.g. `LoyaltyProgramSinceYear`
  ≈ 48.8% missing, `CompetitorOpenSinceYear` ≈ 31.7% missing), which LightGBM
  handles natively without imputation.
- `AppSessions` exists only in training data and was **excluded** to avoid a
  train/test feature mismatch.
- **`IsOpen = 0` implies `OrderVolume = 0`** in the historical data (166,269
  closed-hub rows, all zero; only 54 of 804,110 open-hub rows are zero). This
  relationship is enforced directly in the pipeline rather than left for the
  model to learn: closed-day predictions are hard-set to 0, and closed days
  are excluded from the lag history so they don't distort weekly statistics.

## 3. Feature engineering (51 features)

- **Lags:** 1, 2, 3, 7, 14, 21, 28, 35, 42, 49, 56 days (extended from an
  initial 9-lag set to 11 to support an 8-week seasonal baseline).
- **Same-week statistics:** mean/median/std of the 4 most recent weekly
  lags (7/14/21/28) **and** the 8 most recent (7…56), giving a more stable
  weekday-specific demand estimate than any single lag.
- **Trend:** `trend_4v4` = (mean of the 4 most recent weekly lags) − (mean
  of weeks 5–8 back), capturing whether a hub's demand is rising or falling
  over the past two months.
- **Rolling windows:** trailing 7/14/28-day mean and std, computed causally
  from the full history via a shifted cumulative-sum lookup (so it stays
  correct as recursive predictions are appended during forecasting).
- **Hub expanding baseline:** all-history-to-date mean and std of
  log-demand per hub, using the same cumulative-sum machinery — a stable
  long-run baseline independent of any single lag window.
- **Open-rate context:** trailing 28-day fraction of days the hub was open
  (from `IsOpen`/planned schedule only, no target leakage) — useful signal
  around reopenings after closures.
- **Calendar:** year, month, day-of-month, week-of-year, day-of-year,
  weekend flag, plus cyclic encodings (`weekday_sin/cos`,
  `dayofyear_sin/cos`).
- **Operational:** `IsOpen`, `PromoActive`, `RegionalHoliday`,
  `SchoolClosureFlag`, and interaction terms `promo_open`, `holiday_open`.
- **Hub metadata:** format, assortment tier, competitor distance/tenure,
  loyalty-program status and tenure. `HubID` itself is **not** used as a raw
  numeric feature — hub-specific behavior is instead learned through the
  lag/rolling/expanding features and metadata.

## 4. Model

LightGBM regressor on **`log1p(OrderVolume)`** (RMSLE-aligned), predictions
converted back with `expm1` and clipped at zero.

```
LGBMRegressor(
    objective="regression", learning_rate=0.06, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1
)
```

**Random seeds = [42, 43, 44]**, fixed for NumPy and LightGBM and disclosed
in the notebook. The pipeline fits a model at each of the three seeds and,
on the validation window, checks whether averaging all three ("ensemble")
beats the best single seed; whichever wins is used for the final test
forecast. In this run the single best-tuned seed (42) beat the 3-seed
average (0.1092 vs. 0.1096 validation RMSLE), so the final submission uses
that one model rather than paying the extra training cost of an ensemble
that didn't help.

## 5. Validation methodology

An initial single-step validation (using true lag values throughout the
holdout period) gave an overly optimistic RMSLE of ~0.129 — this leaks future
actuals into the lag features, which is not available at real forecast time.

The validation was corrected to match the actual test-time procedure: a
**leakage-safe, recursive 42-day holdout** (train through 2015-05-08,
forecast 2015-05-09 through 2015-06-19 one day at a time, feeding each day's
prediction back in as history for subsequent lags — exactly as the true test
window must be forecast). A naive hub-by-weekday median baseline was also
computed on the same window as a sanity check.

| Model | Validation RMSLE (all rows) |
|---|---|
| Hub × weekday median baseline | 0.182 |
| LightGBM v1 (9 lags, 38 features), recursive, 600 trees | 0.120 |
| LightGBM v2 (51 features), recursive, 400 trees | 0.110 |
| LightGBM v2, recursive, **600 trees (chosen)** | **0.1092** |
| LightGBM v2, recursive, 800 trees | 0.1092 |
| LightGBM v2, 3-seed ensemble at 600 trees | 0.1096 |

Tree count was selected by comparing recursive validation RMSLE, not
training loss, since training loss on single-step data does not reflect
compounding recursive error. The added rolling-window, 8-week seasonality,
trend, and hub-expanding-baseline features (Section 3) were the main driver
of the v1→v2 improvement (0.120 → 0.109); the 3-seed ensemble was tested
but did not improve on the single-seed result on this validation window,
so it was not used for the final forecast.

## 6. Test-time forecasting

The final model is refit on the full training history (through 2015-06-19)
and forecasts the 42-day test window **recursively**: each day is predicted,
`IsOpen = 0` hubs are forced to 0, and the day's prediction is appended to
the history matrix before generating the next day's lag features. This
mirrors the validation procedure exactly, so the reported validation score
is a realistic estimate of test performance.

## 7. Key findings

- **`same_week_mean_8`, `same_week_mean_4`, and `PromoActive`** are the top
  three features by gain, followed closely by `lag_14`, `lag_1`, and the new
  rolling-mean features — same-weekday history and recent demand dominate,
  and widening the seasonal window from 4 to 8 weeks measurably improved
  both feature importance and validation RMSLE.
- Recursive (leakage-safe) validation RMSLE improved from **0.120 (v1, 38
  features) to 0.109 (v2, 51 features)** — roughly a 9% reduction — driven
  by the rolling-window, extended same-week, trend, and hub-expanding-
  baseline features described in Section 3, not by additional trees or a
  model ensemble.
- A 3-seed ensemble was evaluated on the same recursive validation window
  and did **not** beat the single best-tuned seed (0.1096 vs. 0.1092),
  suggesting the model's error at this stage is dominated by bias
  (missing signal) rather than variance — further feature work is likely
  to help more than further ensembling.
- Both v1 and v2 recursive RMSLE remain meaningfully better than the naive
  hub-by-weekday median baseline (0.182).
- Average demand varies by an order of magnitude across hubs (hub means
  range roughly 2,200–20,700), confirming that a single global demand level
  is inadequate and hub-specific lag/metadata features are necessary.
- Weekday seasonality is strong: open-hub mean demand ranges from ~5,886
  (Saturday) to ~8,216 (Monday), which cyclic and same-week features
  capture directly.

## 8. Reproducibility

The notebook runs end-to-end with no manual intervention: data load →
feature engineering → leakage-safe recursive validation → tree-count
selection → final fit on full history → recursive 42-day test forecast →
`submission.csv`. Seed 42 is fixed for both NumPy and LightGBM.
