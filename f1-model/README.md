# Formula 1 Pre-Race Prediction

**Status: Completed - final pre-race model.**

Aziz Mohammad | Mechanical Engineering, University of Illinois Chicago.

A reproducible motorsport analytics project that combines qualifying observations,
historical results, regression, retirement probabilities and full-field ranking.
It includes three seasons of derived data (2023-2025), model selection on an earlier
season, a separate final-stage test season, plots, an interactive portfolio results
explorer, saved-model inference and preparation of future-race inputs.

The completed scope is **pre-race prediction**. Live telemetry, pit strategy,
tire degradation, DRS and engine-specific failure physics are not implemented.
The original single-season prototype remains in [PHASE1.md](PHASE1.md),
`f1_pipeline.py` and `outputs/`; final-stage results live in `final_outputs/`.

## Run the Final Model

From this directory, with Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Train, validate and evaluate using included historical data; no network needed.
python final_pipeline.py train
python -m pytest -q

# Replay the held-out Abu Dhabi 2025 grid using the evaluation model.
python final_pipeline.py predict \
  --model final_outputs/evaluation_model.joblib \
  --input final_outputs/example_prerace_grid.csv \
  --output final_outputs/example_forecast.csv
```

The example is an explicitly historical replay, not a forecast of an upcoming GP.
Its CSV contains pre-race features only. The actual run output is
[final_outputs/summary.txt](final_outputs/summary.txt). Full scores, data hashes,
model choices, training periods and limitations are in
[final_outputs/metrics.json](final_outputs/metrics.json). Exact installed package
versions for the recorded run are in `requirements-lock.txt` (Python 3.13).

To refresh the source snapshots through FastF1:

```bash
python final_pipeline.py collect
# Cache-only version, after the sessions have already been downloaded:
python final_pipeline.py collect --offline
```

FastF1 loads qualifying laps/weather and race classification, without high-frequency
car telemetry. It caches locally and retries failures in offline mode. Missing race
results fall back to Jolpica's Ergast-compatible API. A missing race stops collection
rather than silently corrupting season history. Missing qualifying features are
logged and subsequently imputed. Per-season source provenance is included in `data/`.

## Forecast an Upcoming Race

Once qualifying and the final starting grid are available, supply a confirmed
grid CSV with one row per qualifying entrant and columns `DriverId,GridPosition`.
Use FastF1 IDs such as `norris` or `max_verstappen`. A grid position of `0` means
a pit-lane start. Do not substitute qualifying classification for the confirmed
grid because penalties can change starting slots.

```bash
python final_pipeline.py prepare --year 2026 --round 1 \
  --grid confirmed_grid.csv --output upcoming_features.csv
python final_pipeline.py predict \
  --model final_outputs/deployment_model.joblib \
  --input upcoming_features.csv --output upcoming_forecast.csv
```

The year/round above is syntax guidance, not a claim about which event is upcoming.
`prepare` loads only the target race's **Q** session and earlier completed races.
It never loads the target race's race session. It rejects missing/duplicate grid
entries and invalid grid slots. Changes to the entry list after qualifying require
manual feature preparation rather than silently guessing driver/team metadata.

`predict` requires the complete field and returns:

- `ConditionalPosition`: predicted position if the driver finishes.
- `RetirementProbability`: probability of retirement conditional on starting.
- `ExpectedPositionScore`: fixed heuristic combining the preceding outputs.
- `PredictedPosition`: unique positions 1 through the supplied field size.
- `GridBaselineRank` and `PositionOnlyRank`: comparison orderings.

All output rows retain their input features. Outcomes such as final position,
points and finish status are never model inputs. Forecast files need `Season`,
`Round`, `FieldSize`, and every feature listed below. No outcome columns are needed.
Only load trusted joblib files: their pickle format can execute Python code.

## Validation and Deployment

1. Fit position-model candidates (Random Forest, Gradient Boosting, Linear
   Regression) on **2023 finishers**. Select the lowest-MAE candidate on **2024**.
2. Fit retirement candidates (Random Forest and regularized Logistic Regression)
   on eligible **2023 starters**. Select the lowest-Brier candidate on **2024**.
3. Refit the selected models on **2023-2024**, then evaluate on the full **2025**
   season. The resulting `evaluation_model.joblib` is frozen for historical replay.
4. After exporting test metrics, refit the selected model types on **2023-2025**
   and save `deployment_model.joblib` for future predictions. Reported holdout
   scores do not apply to this later refit as an independent evaluation.

This replaces the approximate 70/20/10 prototype with explicit season boundaries.
Preprocessing and temperature-outlier limits are fitted inside each training
partition. No normalization, imputation, model selection or weights use the 2025
test outcomes. Evaluation predicts one GP at a time: earlier completed 2025 races
may supply historical features for later 2025 races, while model weights stay fixed.

The 2025 data was already explored during Phase 1. It is held out of final-stage
fitting and model selection, but is not a never-inspected external benchmark.
Treat the reported scores as a retrospective evaluation; a fresh future period
would be needed for a prospective performance claim.

Median imputation retains empty numerical columns and adds missingness indicators.
StandardScaler transforms numeric values; one-hot encoding supports unseen
driver/team categories. Outlier limits use one temperature per training race;
validation/test rows are not removed. Random seeds are fixed at 42.

## Features and Definitions

| Feature | Available before race start |
| --- | --- |
| GridPosition | Confirmed race grid; pit-lane designation 0 is imputed for regression and ordered last in grid tiebreaks. |
| DriverExperience | Earlier GP finishes in the current season; no sprint races. |
| TeamReliability | Earlier team finishes / earlier starts, as a percentage. All-cause finish rate, not mechanical reliability alone. |
| CarPerformanceIndex | Team average classified position over five earlier GPs; both cars averaged within each event. |
| DriverRecentPosition | Driver mean classified position over five earlier GP appearances. |
| QualifyingGapPercent | Best non-deleted timed qualifying lap relative to the session's best, in percent. |
| QualifyingPosition | Official qualifying classification, distinct from grid slot. |
| TrackTemperature / AirTemperature | Mean qualifying Celsius observations; forward-fill only within that session in chronological order. |
| TeamId / DriverId | Stable FastF1 identifiers, one-hot encoded. |
| TireCompound | Fastest valid qualifying lap: HARD=1, MEDIUM=2, SOFT=3, INTERMEDIATE=4, WET=5; categorical with Unknown. |
| TrackStatus | Boolean rainfall proxy: Dry, Mixed Rain, Rain or Unknown. Not flag status or measured track wetness. |

Season history resets at each new year. Missing opening-round history is imputed
from training; it is not represented as perfect reliability. Qualifying pace can
be distorted by evolving weather, track conditions and elimination between
segments. The feature is a useful proxy, not a controlled measurement of car pace.

## Retirement and Full-Field Ordering

Finishers include `Finished`, `Lapped`, and Ergast's `+N Lap(s)` statuses.
Retirement training/evaluation excludes DNS, disqualification, withdrawn,
excluded and unknown outcomes. Other non-finishing statuses count as retirement;
these include accidents as well as mechanical failures.

For each complete field, the fixed ranking score is:

```text
score = (1 - p_retirement) * clipped_conditional_position
      + p_retirement * mean_training_retired_position_fraction * field_size
```

Sort ascending, breaking ties by starting grid and then driver ID. This produces
a unique full-field order but is a heuristic, not a calibrated distribution over
positions or a physical race simulation. Total field size and entries must be
known at inference. Full-field evaluation includes every returned race entry,
including DNS/DSQ; those outcomes are not separately predicted by the classifier.

The report compares the combined ordering against **grid order** and
**position-only order**. Retirement Brier/log loss/ROC AUC are compared with a
constant historical retirement-rate baseline. A functioning classifier does not
prove useful predictive skill: a higher Brier score than the constant baseline
means it performed worse on that metric. The website reports this explicitly.

## Evaluation Artifacts

- `data/season-YYYY.csv` and `provenance-YYYY.json`: derived historical features and sources.
- `final_outputs/metrics.json` / `summary.txt`: measured model selection, holdout and baseline scores.
- `final_outputs/test_predictions.csv`: all 2025 fields, actuals, probabilities and rankings.
- `final_outputs/team_metrics.csv`: full-field per-team MAE with sample counts.
- `final_outputs/retirement_calibration.csv`: observed versus predicted rates in equal-frequency bins.
- `final_outputs/*.png`: full-field scatter, residuals, feature importance and calibration.
- `final_outputs/example_prerace_grid.csv`: outcome-free historical replay input.
- `final_outputs/*model.joblib`: local fitted bundles, excluded from Git and regenerated by training.

Run `python render_portfolio.py` after the final experiment to regenerate the case
study and its race explorer data from measured outputs. The tests cover target
leakage, chronology, history, missing values, retirement labels, unique field ranks,
new driver/team categories and the target-race loading boundary.

## Limitations and Sources

Results cover three historical seasons, not future rules or car generations.
Finisher regression is conditional on finishing; full-field ranking is more
difficult and includes outcomes that were excluded from regression. MAE is an
average error, not a confidence interval. Feature coefficients/importances do not
establish causality, and retirement probabilities are not guarantees.

Data: [FastF1](https://docs.fastf1.dev/core.html) and
[Jolpica](https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/results.md).
The included derived tables support reproducibility; underlying data remains
subject to provider terms. This independent educational project is not affiliated
with Formula 1. No synthetic fixtures contribute to published performance.
