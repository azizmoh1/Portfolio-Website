#!/usr/bin/env python3
"""Final pre-race model: season holdout, retirement risk and full-field ordering.

Fit candidates on 2023, select on 2024, refit on 2023-24, evaluate once on 2025.
Only after evaluation, refit the deployment bundle on all available seasons.
No live telemetry or current-race outcome is consumed at prediction time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import f1_pipeline as base
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
NUMERIC = base.NUMERIC + ["DriverRecentPosition", "QualifyingGapPercent", "QualifyingPosition"]
CATEGORICAL = ["TeamId", "DriverId", "TireCompound", "TrackStatus"]
FEATURES = NUMERIC + CATEGORICAL
NON_RETIREMENT = {"Disqualified", "Did not start", "Did not qualify", "Withdrawn", "Withdrew",
                  "Excluded", "Unknown", "Pending", "", "nan"}
SEASONS = {2023: 22, 2024: 24, 2025: 24}


def feature_frame(data):
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}. Use final_pipeline.py collect/prepare.")
    frame = data[FEATURES].copy()
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame.loc[frame.GridPosition.le(0), "GridPosition"] = np.nan
    for col in CATEGORICAL:
        frame[col] = frame[col].fillna("Unknown").astype(str)
    return frame


def transformer():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                       keep_empty_features=True)),
                              ("scale", StandardScaler())]), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def candidate(kind, name):
    if kind == "position":
        estimator = {
            "random_forest": RandomForestRegressor(n_estimators=200, max_depth=15,
                min_samples_split=5, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
            "linear_regression": LinearRegression(),
        }[name]
    else:
        estimator = {
            "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                min_samples_leaf=8, random_state=42, n_jobs=-1),
            "logistic_regression": LogisticRegression(C=.1, max_iter=3000, random_state=42),
        }[name]
    return Pipeline([("preprocess", transformer()), ("model", estimator)])


def retirement_data(data):
    eligible = data[~data.Status.fillna("Unknown").isin(NON_RETIREMENT)].copy()
    labels = (~eligible.Status.map(base.finished)).astype(int)
    return eligible, labels


def finishers(data):
    return data[data.Status.map(base.finished) & data.FinalPosition.between(1, 20)].copy()


def regression_metrics(actual, prediction):
    return {"MAE": float(mean_absolute_error(actual, prediction)),
            "MSE": float(mean_squared_error(actual, prediction)),
            "R2": float(r2_score(actual, prediction)), "n": len(actual)}


def probability_metrics(actual, probability):
    return {"Brier": float(brier_score_loss(actual, probability)),
            "LogLoss": float(log_loss(actual, probability, labels=[0, 1])),
            "ROC_AUC": float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None,
            "n": len(actual), "retirements": int(np.sum(actual))}


def fit_bundle(data, position_name, retirement_name):
    clean_train, outliers = base.filter_training_outliers(data)
    completed = finishers(clean_train)
    eligible, labels = retirement_data(clean_train)
    if labels.nunique() != 2:
        raise ValueError("Retirement training requires both finishers and retirements")
    position = candidate("position", position_name)
    position.fit(feature_frame(completed), completed.FinalPosition)
    retirement = candidate("retirement", retirement_name)
    retirement.fit(feature_frame(eligible), labels)
    retired = eligible.loc[labels.eq(1)]
    return {"position_model": position, "retirement_model": retirement,
            "position_name": position_name, "retirement_name": retirement_name,
            "retirement_rate": float(labels.mean()),
            "retired_position_fraction": float((retired.FinalPosition / retired.FieldSize).mean()),
            "grid_median": float(completed.GridPosition[completed.GridPosition.gt(0)].median()),
            "features": FEATURES, "training_seasons": sorted(data.Season.unique().tolist()),
            "outliers": outliers, "schema_version": 1}


def rank_field(data, bundle):
    """Unique 1..N order per complete field; blend is fixed before test evaluation."""
    for col in ["Season", "Round", "FieldSize", "DriverId", *FEATURES]:
        if col not in data:
            raise ValueError(f"Prediction input needs {col}")
    if data.empty or data.duplicated(["Season", "Round", "DriverId"]).any():
        raise ValueError("Empty field or duplicate driver entries")
    if data[["Season", "Round", "FieldSize", "DriverId"]].isna().any().any():
        raise ValueError("Race identifiers, driver IDs and field size cannot be missing")
    output = data.reset_index(drop=True).copy()
    x = feature_frame(output)
    output["ConditionalPosition"] = bundle["position_model"].predict(x)
    output["RetirementProbability"] = bundle["retirement_model"].predict_proba(x)[:, 1]
    output["PredictedPosition"] = 0
    output["GridBaselineRank"] = 0
    output["PositionOnlyRank"] = 0
    for _, field in output.groupby(["Season", "Round"], sort=True):
        n = len(field)
        if n < 2 or not pd.to_numeric(field.FieldSize, errors="coerce").eq(n).all():
            raise ValueError("Supply the complete field: FieldSize must equal the number of drivers per race")
        p = field.RetirementProbability
        conditional = field.ConditionalPosition.clip(1, n)
        # Expected position is a ranking score, not a calibrated distribution over finishing slots.
        score = (1-p)*conditional + p*bundle["retired_position_fraction"]*n
        output.loc[field.index, "ExpectedPositionScore"] = score
        order = field.assign(score=score, grid=field.GridPosition.where(field.GridPosition.gt(0), n+1))
        for column, sort_columns in [("PredictedPosition", ["score", "grid", "DriverId"]),
                                     ("GridBaselineRank", ["grid", "DriverId"]),
                                     ("PositionOnlyRank", ["ConditionalPosition", "grid", "DriverId"])]:
            idx = order.sort_values(sort_columns, kind="stable").index
            output.loc[idx, column] = np.arange(1, n+1)
    return output.sort_values(["Season", "Round", "PredictedPosition"])


def ranking_metrics(predictions, column):
    valid = predictions[predictions.FinalPosition.notna() & predictions.FinalPosition.gt(0)]
    metrics = regression_metrics(valid.FinalPosition, valid[column])
    correlations, podiums, winners = [], [], []
    for _, race in valid.groupby(["Season", "Round"]):
        correlations.append(float(race.FinalPosition.corr(race[column], method="spearman")))
        podiums.append(len(set(race.loc[race.FinalPosition.le(3), "DriverId"]) &
                           set(race.loc[race[column].le(3), "DriverId"])) / 3)
        winners.append(bool((race.FinalPosition.eq(1) & race[column].eq(1)).any()))
    metrics.update({"MeanRaceSpearman": float(np.mean(correlations)),
                    "PodiumRecall": float(np.mean(podiums)),
                    "WinnerAccuracy": float(np.mean(winners)), "races": len(correlations)})
    return metrics


def load_data(directory):
    frames = []
    for year in SEASONS:
        path = Path(directory) / f"season-{year}.csv"
        if not path.exists():
            raise ValueError(f"Missing {path}. Run collect first.")
        frame = pd.read_csv(path, dtype={c: str for c in CATEGORICAL})
        feature_frame(frame)
        if set(frame.Season) != {year} or frame.Round.nunique() != SEASONS[year]:
            raise ValueError(f"Expected the complete {year} season in {path}")
        if frame.duplicated(["Season", "Round", "DriverId"]).any():
            raise ValueError(f"Duplicate entries in {path}")
        if frame[["DriverId", "TeamId", "Status", "FinalPosition"]].isna().any().any():
            raise ValueError(f"Incomplete identity/outcome metadata in {path}")
        if not frame.FinalPosition.between(1, 20).all():
            raise ValueError(f"Invalid outcome positions in {path}")
        for _, race in frame.groupby("Round"):
            if not race.FieldSize.eq(len(race)).all():
                raise ValueError(f"Incomplete race field in {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["Season", "Round", "DriverId"])


def plot_final(predictions, bundle, calibration, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predictions.FinalPosition, predictions.PredictedPosition, alpha=.22, s=35, color="#2563eb")
    ax.plot([1, 20], [1, 20], "--", color="#64748b")
    ax.set(xlabel="Actual classified position", ylabel="Predicted full-field rank",
           title="2025 season holdout | full-field prediction", xticks=range(2, 21, 2), yticks=range(2, 21, 2))
    fig.tight_layout(); fig.savefig(output / "full_field.png", dpi=180); plt.close(fig)
    names = bundle["position_model"]["preprocess"].get_feature_names_out()
    estimator = bundle["position_model"]["model"]
    if hasattr(estimator, "feature_importances_"):
        values, label = estimator.feature_importances_, "Impurity-based importance (not causal)"
    else:
        values, label = np.abs(estimator.coef_), "Absolute coefficient (scaled numeric features; not causal)"
    importance = pd.Series(values, index=names).sort_values(ascending=False)
    importance.to_csv(output / "feature_importances.csv", header=["importance"])
    top = importance.head(10).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([n.split("__", 1)[-1] for n in top.index], top.values, color="#2563eb")
    ax.set(title=f"Position model | {bundle['position_name'].replace('_', ' ')}", xlabel=label)
    fig.tight_layout(); fig.savefig(output / "feature_importances.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect calibration")
    ax.plot(calibration.PredictedRisk, calibration.ObservedRate, "o-", color="#2563eb", label="2025 test bins")
    ax.set(xlabel="Mean predicted retirement probability", ylabel="Observed retirement rate",
           title="Retirement calibration | equal-frequency bins", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(output / "retirement_calibration.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(predictions.PredictedPosition-predictions.FinalPosition, bins=np.arange(-20.5, 21), color="#2563eb")
    ax.axvline(0, color="#ef4444", linestyle="--")
    ax.set(xlabel="Predicted rank minus actual position", ylabel="Driver entries", title="Full-field test errors")
    fig.tight_layout(); fig.savefig(output / "residuals.png", dpi=180); plt.close(fig)


def train_final(data_dir, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    train, val, test = [data[data.Season.eq(year)].copy() for year in SEASONS]
    filtered, _ = base.filter_training_outliers(train)
    train_finished, val_finished = finishers(filtered), finishers(val)
    pos_validation, risk_validation = {}, {}
    for name in ["random_forest", "gradient_boosting", "linear_regression"]:
        model = candidate("position", name)
        model.fit(feature_frame(train_finished), train_finished.FinalPosition)
        pred = model.predict(feature_frame(val_finished))
        pos_validation[name] = regression_metrics(val_finished.FinalPosition, pred)
    eligible, labels = retirement_data(filtered)
    val_eligible, val_labels = retirement_data(val)
    for name in ["random_forest", "logistic_regression"]:
        model = candidate("retirement", name)
        model.fit(feature_frame(eligible), labels)
        prob = model.predict_proba(feature_frame(val_eligible))[:, 1]
        risk_validation[name] = probability_metrics(val_labels, prob)
    position_name = min(pos_validation, key=lambda name: pos_validation[name]["MAE"])
    retirement_name = min(risk_validation, key=lambda name: risk_validation[name]["Brier"])
    print(f"Selected on 2024: position={position_name}, retirement={retirement_name}")
    bundle = fit_bundle(data[data.Season.lt(2025)], position_name, retirement_name)
    predictions = rank_field(test, bundle)
    completed = finishers(predictions)
    conditional_metrics = regression_metrics(completed.FinalPosition, completed.ConditionalPosition)
    eligible, actual = retirement_data(predictions)
    probability = eligible.RetirementProbability.to_numpy()
    risk_metrics = probability_metrics(actual, probability)
    risk_baseline = probability_metrics(actual, np.full(len(actual), bundle["retirement_rate"]))
    calibration = pd.DataFrame({"PredictedRisk": probability, "ObservedRate": actual.to_numpy()})
    calibration["Bin"] = pd.qcut(calibration.PredictedRisk, q=5, duplicates="drop")
    calibration = calibration.groupby("Bin", observed=True).agg(
        PredictedRisk=("PredictedRisk", "mean"), ObservedRate=("ObservedRate", "mean"), N=("ObservedRate", "size")).reset_index(drop=True)
    rankings = {name: ranking_metrics(predictions, col) for name, col in
                [("combined", "PredictedPosition"), ("grid_baseline", "GridBaselineRank"), ("position_only", "PositionOnlyRank")]}
    teams = predictions.assign(AbsoluteError=abs(predictions.PredictedPosition-predictions.FinalPosition)).groupby("TeamName").agg(MAE=("AbsoluteError", "mean"), N=("AbsoluteError", "size"))
    report = {
        "status": "Completed - Final pre-race model", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Pre-race only: conditional position, retirement probability, unique field ordering",
        "selection_train": 2023, "selection_validation": 2024, "evaluation_training": [2023, 2024], "test_season": 2025,
        "deployment_training": [2023, 2024, 2025],
        "position_model": position_name, "retirement_model": retirement_name,
        "position_validation": pos_validation, "retirement_validation": risk_validation,
        "conditional_position_test": conditional_metrics, "retirement_test": risk_metrics,
        "retirement_constant_baseline": risk_baseline, "full_field_test": rankings,
        "team_test": {name: {"MAE": float(r.MAE), "n": int(r.N)} for name, r in teams.iterrows()},
        "data": {str(year): {"rows": len(data[data.Season.eq(year)]), "races": data[data.Season.eq(year)].Round.nunique(),
                 "finishers": len(finishers(data[data.Season.eq(year)])),
                 "sha256": hashlib.sha256((Path(data_dir)/f"season-{year}.csv").read_bytes()).hexdigest()} for year in SEASONS},
        "features": FEATURES, "outlier_filter": bundle["outliers"],
        "missing_numeric_fraction": {str(year): {c: float(v) for c, v in data.loc[data.Season.eq(year), NUMERIC].isna().mean().items()} for year in SEASONS},
        "prediction_cutoff": "After qualifying and confirmed grid publication, before race start",
        "history_policy": "One upcoming race at a time; earlier 2025 outcomes may update rolling features, never model weights",
        "rank_policy": "Sort (1-p_retire)*conditional_position + p_retire*training_mean_retired_position_fraction*field_size; ties resolved by grid then driver ID",
        "versions": {p: base.version(p) for p in ["fastf1", "pandas", "numpy", "scikit-learn"]},
        "limitations": ["Retirement includes crashes and mechanical failures; DNS and DSQ are excluded from classifier evaluation.",
            "2025 was explored during Phase 1; it is held out of final-stage model selection, not a never-inspected external benchmark.",
            "Full-field evaluation includes every returned entry, including DNS/DSQ; these outcomes are not separately modeled.",
            "Unique ordering is an expected-position heuristic, not a probabilistic race simulation.",
            "No claim that retirement probabilities are useful unless they beat the constant-rate baseline.",
            "Three historical seasons do not establish performance under future regulation changes.",
            "Qualifying pace/compound/rain are proxies; no live strategy, tire degradation, DRS or pit-loss model."]}
    (output/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    predictions.to_csv(output/"test_predictions.csv", index=False)
    calibration.to_csv(output/"retirement_calibration.csv", index=False)
    teams.to_csv(output/"team_metrics.csv")
    joblib.dump(bundle, output/"evaluation_model.joblib")
    plot_final(predictions, bundle, calibration, output)
    # Deployment refit happens after the held-out metrics and artifacts are frozen.
    joblib.dump(fit_bundle(data, position_name, retirement_name), output/"deployment_model.joblib")
    sample = test[test.Round.eq(24)][["Season", "Round", "RaceName", "Driver", "TeamName", "FieldSize", *FEATURES]]
    sample.to_csv(output/"example_prerace_grid.csv", index=False)
    summary = ["Final pre-race pipeline complete.",
        f"Selection: 2023 train / 2024 validation; test: full 2025 season ({len(test)} entries).",
        f"Position model: {position_name}; retirement model: {retirement_name}.",
        f"Finisher-only position MAE: {conditional_metrics['MAE']:.3f}; R2: {conditional_metrics['R2']:.3f}.",
        f"Full-field rank MAE: {rankings['combined']['MAE']:.3f}; grid baseline: {rankings['grid_baseline']['MAE']:.3f}; position-only: {rankings['position_only']['MAE']:.3f}.",
        f"Retirement Brier: {risk_metrics['Brier']:.4f}; constant baseline: {risk_baseline['Brier']:.4f}; ROC AUC: {risk_metrics['ROC_AUC']:.3f}.",
        "Deployment model refitted on 2023-2025 after evaluation. No test metrics reported for that refit.", *report["limitations"]]
    (output/"summary.txt").write_text("\n".join(summary)+"\n")
    print("\n".join(summary))
    return report


def prepare_grid(year, number, grid_path, cache, offline=False):
    """Build future-race input from Q, confirmed grid and earlier results only."""
    if number < 1:
        raise ValueError("Round must be positive")
    grid = pd.read_csv(grid_path, dtype={"DriverId": str})
    if not {"DriverId", "GridPosition"}.issubset(grid):
        raise ValueError("Confirmed grid CSV needs DriverId and GridPosition")
    if grid.DriverId.isna().any() or grid.DriverId.duplicated().any():
        raise ValueError("Grid driver IDs must be present and unique")
    grid.GridPosition = pd.to_numeric(grid.GridPosition, errors="coerce")
    if grid.GridPosition.isna().any() or (grid.GridPosition < 0).any() or (grid.GridPosition > len(grid)).any():
        raise ValueError("Grid positions must be 0 (pit lane) or a valid starting slot")
    slots = grid.loc[grid.GridPosition.gt(0), "GridPosition"]
    if slots.duplicated().any() or not slots.mod(1).eq(0).all():
        raise ValueError("Positive grid slots must be unique integers")
    q = base.load_and_cache_session(year, number, "Q", cache, offline)
    if set(q.results.DriverId) != set(grid.DriverId):
        raise ValueError("Confirmed grid must contain every qualifying entry exactly once")
    if number > 1:
        history, _ = base.build_dataset(year, list(range(1, number)), cache, offline)
    else:
        history = pd.DataFrame(columns=["Season", "Round", "DriverId", "TeamId", "FinalPosition", "Status", "Finished"])
    entries = q.results.copy().drop(columns=["GridPosition"], errors="ignore")
    entries = entries.merge(grid[["DriverId", "GridPosition"]], on="DriverId", validate="one_to_one")
    entries["Position"], entries["Points"], entries["Status"] = np.nan, np.nan, "Pending"
    result = base.extract_features(entries, q, history, year, number, q.event["EventName"])
    return result[["Season", "Round", "RaceName", "Driver", "TeamName", "FieldSize", *FEATURES]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--data-dir", type=Path, default=ROOT/"data")
    collect.add_argument("--cache-dir", type=Path, default=ROOT/"cache")
    collect.add_argument("--offline", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--data-dir", type=Path, default=ROOT/"data")
    train.add_argument("--output-dir", type=Path, default=ROOT/"final_outputs")
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--year", type=int, required=True)
    prepare.add_argument("--round", type=int, required=True)
    prepare.add_argument("--grid", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, default=ROOT/"cache")
    prepare.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.command == "collect":
        args.data_dir.mkdir(parents=True, exist_ok=True)
        for year, count in SEASONS.items():
            frame, provenance = base.build_dataset(year, range(1, count+1), args.cache_dir, args.offline)
            frame.to_csv(args.data_dir/f"season-{year}.csv", index=False)
            (args.data_dir/f"provenance-{year}.json").write_text(json.dumps(provenance, indent=2)+"\n")
    elif args.command == "train":
        train_final(args.data_dir, args.output_dir)
    elif args.command == "prepare":
        frame = prepare_grid(args.year, args.round, args.grid, args.cache_dir, args.offline)
        args.output.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(args.output, index=False)
    else:
        # Never load untrusted joblib files; they use Python pickle.
        bundle = joblib.load(args.model)
        frame = pd.read_csv(args.input, dtype={c: str for c in CATEGORICAL})
        out = rank_field(frame, bundle)
        args.output.parent.mkdir(parents=True, exist_ok=True); out.to_csv(args.output, index=False)
        print(out[["DriverId", "PredictedPosition", "RetirementProbability"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (base.DataLoadError, ValueError, KeyError, OSError) as exc:
        raise SystemExit(f"Pipeline stopped: {exc}") from exc
