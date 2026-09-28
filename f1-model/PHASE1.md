# Formula 1 Race Prediction

**Status: Completed - Phase 1 (data engineering and baseline prediction).**

Aziz Mohammad | Mechanical Engineering, University of Illinois Chicago.

A reproducible Python pipeline connecting motorsport engineering questions to
historical data and machine learning. Predicts a driver's finishing position
given qualifying observations, the published starting grid, and prior race results.
Includes FastF1 ingestion, cache/offline support, feature engineering, model
comparison, held-out evaluation, exportable figures, saved-model inference, and tests.

This is an educational retrospective model conditional on a driver finishing.
It does not predict retirements or run a live telemetry feed. Tire degradation,
DRS effectiveness, pit-loss physics and engine-specific failure models remain
future extensions. Completion refers to the implemented Phase 1 scope.

## Quick Start

From this `f1-model` directory (Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python f1_pipeline.py train
python -m pytest -q
```

The default run loads all 24 rounds of 2025. The initial download takes several
minutes; subsequent runs reuse the local cache. It does not download car-position
or high-frequency telemetry streams because this baseline does not consume them.
For the exact package versions used for the published run, install
`requirements-lock.txt` (recorded on Python 3.13).

```bash
# Small five-race demonstration, including Australia, Bahrain and Saudi Arabia.
python f1_pipeline.py train --year 2025 --rounds 1 2 3 4 5 --output-dir outputs-demo

# Reproduce the published experiment from the included feature table, no network.
python f1_pipeline.py train --dataset outputs/dataset.csv --output-dir outputs-reproduced

# Use only your local FastF1/Jolpica cache (must already contain the sessions).
python f1_pipeline.py train --offline --output-dir outputs-offline

# Evaluate points instead of finishing positions; race points are read from results.
python f1_pipeline.py train --dataset outputs/dataset.csv --target Points_Scored --output-dir outputs-points
```

The experiment prints real MAE, MSE, R2 and per-team MAE with sample counts.
[`outputs/summary.txt`](outputs/summary.txt) is the actual example output,
not an invented performance illustration. [`outputs/metrics.json`](outputs/metrics.json)
contains split membership, data sources, missingness, package versions and limitations.

## Feature Contract

Prediction cutoff: after qualifying and the final starting grid are available,
before race start. Historical final grids are taken from results metadata. They
include grid penalties; they are not identical to qualifying classification.

| Feature | Definition |
| --- | --- |
| GridPosition | Published race starting slot; 0 (pit lane/unknown) becomes missing. |
| TeamName | One-hot encoded constructor name; unseen teams are supported. |
| DriverExperience | Grand Prix finishes earlier in the same season; excludes sprints. |
| TeamReliability | Earlier team finishes / earlier team starts, as a percentage. DNS/DNQ/withdrawals do not count as starts. This is all-cause finish rate, not mechanical reliability alone. |
| CarPerformanceIndex | Mean of the team's per-GP average classified positions in its last five earlier GPs, including DNF classification positions. Both cars are averaged within each GP. |
| TireCompound | Compound of the driver's fastest non-deleted timed qualifying lap. HARD=1, MEDIUM=2, SOFT=3, INTERMEDIATE=4, WET=5, otherwise Unknown. One-hot encoded, not treated as an ordered number. |
| TrackTemperature | Mean qualifying track-surface temperature in Celsius, after forward-fill in time order within that session. |
| AirTemperature | Mean qualifying ambient temperature in Celsius, with the same session-only forward-fill. |
| TrackStatus | Qualifying rainfall proxy: Dry, Mixed Rain, Rain, Unknown. FastF1 rainfall is boolean. This is neither flag status nor a measured wetness estimate. |

Current-race final position, points and finish status never enter the feature
matrix. The loader traverses every earlier round even if only a later subset is
requested, so season history stays complete. Stable driver/constructor IDs join
history; extraction precedes the current race's history update. History resets
by season. No previous history is represented as missing, not perfect reliability.

## Validation Design

Whole races are split chronologically, approximately 70% / 20% / 10%.
For 24 rounds this is **16 training, 5 validation, 3 test races**. For 3-5 races,
one race each is reserved for validation and test, so the ratio is necessarily
coarser. A random driver-level team-stratified split would mix the same races
across partitions and can fail when a 10% test set has fewer rows than teams.
Team coverage is reported rather than guaranteed by leaking event information.

Evaluation is walk-forward: the model is fitted on training races only, while
each later race's features can use results of earlier completed validation/test
races. This simulates predicting one upcoming GP at a time. It is not a forecast
of the whole remaining season made at the end of round 16.

Median imputation (including missing-value indicators), StandardScaler and
OneHotEncoder are fitted on training rows only. Entirely missing numerical
columns are retained with a zero fallback and missing indicator. Qualifying
temperature outliers beyond +/-3 training standard deviations are removed from
training only. Validation/test distributions and labels remain untouched.

Random Forest is the prespecified primary model (100 trees, depth 15,
minimum split 5, seed 42). Gradient Boosting and Linear Regression are compared
on validation, without selecting a model or tuning hyperparameters on test data.
A grid-position baseline measures whether the added complexity is useful.
The points experiment uses the training target mean as its baseline.

MAE is average absolute error, **not** a confidence interval or a promise that
each prediction lies within that range. R2 can be negative. Per-team metrics
are noisy with only a few held-out driver entries. Feature importances are
impurity-based and can favor continuous/high-cardinality features; they are not
causal evidence. Single-season results are not enough to establish performance
across years, circuits, regulation changes or adverse weather.

## Saved-Model Inference

Training writes `outputs/model.joblib` (excluded from Git; regenerate locally).
Only load trusted joblib files because they use Python pickle.

```bash
python f1_pipeline.py predict --model outputs/model.joblib --input examples/prerace_features.csv --output outputs/example_predictions.csv
```

The example input is explicitly hypothetical and demonstrates the schema; it is
not included in training or published metrics. Replace feature values with
observations available at the stated prediction cutoff. Output retains input
columns and adds a continuous `Prediction` column. Independent regression can
produce ties and does not enforce a unique 1-20 finishing order. Points predictions
are also continuous. No ground-truth outcome columns are needed for inference.

## Outputs

- `dataset.csv`: all extracted driver rows including DNFs, for audit/reproduction.
- `splits.csv`: exact eligible driver/race partition assignments.
- `metrics.json` and `summary.txt`: real validation/test results and provenance.
- `test_predictions.csv`: held-out labels, predictions, baseline and absolute errors.
- `pred_vs_actual.png`: scatter plot with the perfect-prediction line.
- `feature_importances.png` / `.csv`: top 10 chart and full feature importance table.
- `residuals.png`: distribution of test errors.
- `model.joblib`: fitted preprocessing and Random Forest, generated locally.

## Failure Handling and Data Sources

FastF1 enables its local cache before loading a session. A failed load is retried
in offline mode. Missing race results fall back to the
[Jolpica Ergast-compatible results API](https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/results.md),
with its own JSON cache. If neither source has race results, the run stops with
the failed round rather than silently computing incomplete historical features.
Missing qualifying/weather/tire observations are logged and imputed; unknown
rain conditions are not mislabeled as dry. Sources and availability are recorded.
An offline run cannot conjure sessions that were never cached.

Session/result/weather semantics follow [FastF1 documentation](https://docs.fastf1.dev/core.html).
This independent educational project is not affiliated with Formula 1. The included
derived feature table is for reproducibility; underlying data remains subject to
its providers' terms. No fabricated results are mixed into published metrics.

## Engineering Extensions

The next useful work is multi-season evaluation, a retirement model, and a race
ranking objective. Physics-oriented extensions can then include tire degradation,
DRS speed delta, and circuit-specific pit-lane loss. Stationary pit service time
must be distinguished from total pit-stop race-time loss (entry, lane transit,
service and exit). Those variables need their own prediction cutoff to prevent
using strategy decisions that have not happened yet.
