#!/usr/bin/env python3
"""Formula 1 finishing-position prediction | Aziz Mohammad, UIC Mechanical Engineering.

Completed Phase 1 portfolio pipeline: connect motorsport data
to reproducible data engineering and predictive analytics. Predict at the point
when qualifying and the starting grid are known. Current-race outcomes are labels
only; all rolling statistics use earlier Grands Prix. Physics-based tire wear,
DRS, pit-loss and live-race models are future extensions, not implemented claims.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import fastf1
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOG = logging.getLogger("f1-model")
NUMERIC = ["GridPosition", "DriverExperience", "TeamReliability",
           "CarPerformanceIndex", "TrackTemperature", "AirTemperature"]
CATEGORICAL = ["TeamName", "TireCompound", "TrackStatus"]
FEATURES = NUMERIC + CATEGORICAL
TIRES = {"HARD": "1", "MEDIUM": "2", "SOFT": "3",
         "INTERMEDIATE": "4", "WET": "5"}


class DataLoadError(RuntimeError):
    """An unavailable source must never silently become fabricated race data."""


def finished(status: str) -> bool:
    """Accept finished and lapped finishers; reject retirements, DNS and DSQ."""
    return bool(re.fullmatch(r"Finished|Lapped|\+\d+ Laps?", str(status)))


def load_and_cache_session(year, round_number, kind, cache_dir, offline=False):
    """Use FastF1's two-level cache; retry a failed network load in cache-only mode."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    errors = []
    for cache_only in ([True] if offline else [False, True]):
        fastf1.Cache.offline_mode(cache_only)
        try:
            session = fastf1.get_session(year, round_number, kind)
            session.load(laps=kind == "Q", telemetry=False,
                         weather=kind == "Q", messages=False)
            if session.results.empty:
                raise DataLoadError("No session results returned")
            if kind == "R":
                validate_race_results(session.results)
            return session
        except Exception as exc:
            errors.append(str(exc))
        finally:
            fastf1.Cache.offline_mode(offline)
    raise DataLoadError(f"{year} round {round_number} {kind}: " + " | ".join(errors))


def validate_race_results(results):
    """FastF1 can return a populated entry list while result metadata failed."""
    for column in ["DriverId", "TeamId", "Position", "Status"]:
        if column not in results or results[column].isna().any():
            raise DataLoadError(f"Incomplete race results: missing {column}")
    if results.DriverId.astype(str).str.strip().eq("").any() or results.DriverId.duplicated().any():
        raise DataLoadError("Incomplete race results: driver IDs are empty or duplicated")
    if not pd.to_numeric(results.Position, errors="coerce").gt(0).all():
        raise DataLoadError("Incomplete race results: invalid finishing positions")


def fallback_results(year, round_number, cache_dir, offline=False):
    """Ergast-compatible Jolpica results fallback with an explicit JSON cache."""
    path = Path(cache_dir) / f"jolpica-{year}-{round_number}.json"
    url = f"https://api.jolpi.ca/ergast/f1/{year}/{round_number}/results.json?limit=100"
    if path.exists():
        payload = json.loads(path.read_text())
    elif offline:
        raise DataLoadError(f"No cached Jolpica results for round {round_number}")
    else:
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            payload = response.json()
            races = payload["MRData"]["RaceTable"]["Races"]
            if not races or not races[0]["Results"]:
                raise ValueError("No completed race results")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise DataLoadError(f"Jolpica unavailable: {url}: {exc}") from exc
    race = payload["MRData"]["RaceTable"]["Races"][0]
    rows = [{"DriverId": r["Driver"]["driverId"],
             "Abbreviation": r["Driver"].get("code", r["Driver"]["driverId"]),
             "TeamName": r["Constructor"]["name"],
             "TeamId": r["Constructor"]["constructorId"],
             "GridPosition": r.get("grid"), "Position": r["position"],
             "Points": r["points"], "Status": r["status"]}
            for r in race["Results"]]
    frame = pd.DataFrame(rows)
    validate_race_results(frame)
    return frame, race["raceName"]


def weather_features(weather):
    """Forward-fill chronological samples within qualifying only, in Celsius.

    Rainfall is boolean, not rainfall intensity or FastF1's flag TrackStatus.
    Dry/Mixed Rain/Rain are a coarse weather proxy, not measured track wetness.
    """
    if weather is None or weather.empty:
        return np.nan, np.nan, "Unknown"
    weather = weather.sort_values("Time") if "Time" in weather else weather.copy()
    values = []
    for col in ["TrackTemp", "AirTemp"]:
        series = pd.to_numeric(weather.get(col, pd.Series(dtype=float)), errors="coerce")
        values.append(float(series.ffill().mean()))
    rain = weather.get("Rainfall", pd.Series(dtype=float)).dropna()
    # Mixed rain indicates rain reported during some, but not all, observations.
    condition = "Unknown" if rain.empty else (
        "Dry" if not rain.astype(bool).any() else
        "Rain" if rain.astype(bool).all() else "Mixed Rain")
    return *values, condition


def qualifying_features(session):
    if session is None:
        return (np.nan, np.nan, "Unknown"), {}
    try:
        weather = weather_features(session.weather_data)
    except Exception:
        weather = (np.nan, np.nan, "Unknown")
    compounds = {}
    try:
        # Qualifying compound is known before the race, unlike race stint data.
        laps = session.laps.dropna(subset=["LapTime"]).sort_values("LapTime")
        if "Deleted" in laps:
            laps = laps[~laps["Deleted"].eq(True)]
        for driver, group in laps.groupby("Driver"):
            compounds[driver] = TIRES.get(str(group.iloc[0]["Compound"]), "Unknown")
    except Exception as exc:
        LOG.warning("Qualifying compounds unavailable: %s", exc)
    return weather, compounds


def extract_features(results, qual_session, history, year, round_number, race_name):
    """One row per driver; history is unchanged until the entire race is extracted."""
    weather, compounds = qualifying_features(qual_session)
    pace, qual_positions = {}, {}
    if qual_session is not None:
        try:
            laps = qual_session.laps.dropna(subset=["LapTime"])
            if "Deleted" in laps:
                laps = laps[~laps["Deleted"].eq(True)]
            best = laps.groupby("Driver")["LapTime"].min().dt.total_seconds()
            pace = ((best / best.min() - 1) * 100).to_dict()
            qual_positions = qual_session.results.set_index("Abbreviation")["Position"].to_dict()
        except Exception as exc:
            LOG.warning("Qualifying pace/position unavailable: %s", exc)
    rows = []
    for _, result in results.iterrows():
        driver, team = str(result["DriverId"]), str(result["TeamId"])
        past = history[history["Season"].eq(year) & history["Round"].lt(round_number)]
        driver_past = past[past["DriverId"].eq(driver)]
        team_past = past[past["TeamId"].eq(team)]
        # Average both cars within each GP, then the last five GPs (not five cars).
        team_races = team_past.groupby("Round")["FinalPosition"].mean().tail(5)
        starts = team_past[~team_past["Status"].isin(["Did not start", "Did not qualify", "Withdrawn", "Withdrew"])]
        rows.append({
            "Season": year, "Round": round_number, "RaceName": race_name,
            "DriverId": driver, "Driver": result["Abbreviation"], "TeamId": team,
            "TeamName": result["TeamName"],
            "GridPosition": pd.to_numeric(result["GridPosition"], errors="coerce"),
            "DriverExperience": int(driver_past["Finished"].sum()),
            "TeamReliability": float(starts["Finished"].mean() * 100) if len(starts) else np.nan,
            "CarPerformanceIndex": float(team_races.mean()) if len(team_races) else np.nan,
            "DriverRecentPosition": float(driver_past.sort_values("Round")["FinalPosition"].tail(5).mean()),
            "QualifyingGapPercent": pace.get(result["Abbreviation"], np.nan),
            "QualifyingPosition": qual_positions.get(result["Abbreviation"], np.nan),
            "FieldSize": len(results),
            "TireCompound": compounds.get(result["Abbreviation"], "Unknown"),
            "TrackTemperature": weather[0], "AirTemperature": weather[1],
            "TrackStatus": weather[2],
            "FinalPosition": pd.to_numeric(result["Position"], errors="coerce"),
            "Points_Scored": pd.to_numeric(result["Points"], errors="coerce"),
            "Status": result["Status"], "Finished": finished(result["Status"]),
        })
    return pd.DataFrame(rows)


def build_dataset(year, rounds, cache_dir, offline=False):
    """Load all prior rounds to prevent gaps in season-to-date features."""
    rounds = sorted(set(rounds))
    if not rounds or min(rounds) < 1:
        raise ValueError("Provide positive round numbers")
    history = pd.DataFrame(columns=["Season", "Round", "DriverId", "TeamId",
                                    "FinalPosition", "Status", "Finished"])
    data, provenance = [], []
    for number in range(1, max(rounds) + 1):
        LOG.info("Loading %s round %s", year, number)
        source = "FastF1"
        try:
            race = load_and_cache_session(year, number, "R", cache_dir, offline)
            results, name = race.results.copy(), race.event["EventName"]
        except DataLoadError as exc:
            LOG.warning("%s; trying Jolpica", exc)
            results, name = fallback_results(year, number, cache_dir, offline)
            source = "Jolpica"
        qual = None
        if number in rounds:
            try:
                qual = load_and_cache_session(year, number, "Q", cache_dir, offline)
            except DataLoadError as exc:
                LOG.warning("%s; missing qualifying features will be imputed", exc)
        frame = extract_features(results, qual, history, year, number, name)
        history = frame.copy() if history.empty else pd.concat([history, frame], ignore_index=True)
        if number in rounds:
            data.append(frame)
            provenance.append({"season": year, "round": number, "race": name,
                               "results_source": source, "qualifying_loaded": qual is not None})
    return pd.concat(data, ignore_index=True), provenance


def clean_dataset(df, target="FinalPosition"):
    df = df.copy()
    if df.duplicated(["Season", "Round", "DriverId"]).any():
        raise ValueError("Duplicate driver/race rows")
    for col in NUMERIC + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").astype(str)
    # Zero is a pit-lane/unknown grid designation, never pole position.
    df.loc[df["GridPosition"] <= 0, "GridPosition"] = np.nan
    df = df[df["Finished"].eq(True) & df[target].notna()].copy()
    if target == "FinalPosition":
        df = df[df[target].between(1, 20)]
    return df.sort_values(["Season", "Round", "DriverId"]).reset_index(drop=True)


def split_by_race(df):
    """Approximate 70/20/10, keeping complete races in time order.

    With 24 GPs this is 16/5/3. Small 3-5 GP demonstrations reserve one
    validation and one test GP. Team stratification would mix race outcomes.
    """
    races = list(df[["Season", "Round"]].drop_duplicates().sort_values(
        ["Season", "Round"]).itertuples(index=False, name=None))
    if len(races) < 3:
        raise ValueError("At least three races with finishers are required")
    n_test = max(1, int(np.ceil(len(races) * .1)))
    n_val = max(1, int(round(len(races) * .2)))
    n_train = len(races) - n_val - n_test
    keys = list(zip(df.Season, df.Round))
    groups = [races[:n_train], races[n_train:n_train+n_val], races[n_train+n_val:]]
    return tuple(df.loc[[key in group for key in keys]].copy() for group in groups)


def filter_training_outliers(train):
    """Fit +/-3 sigma on training race temperatures; never filter held-out labels."""
    temperatures = train.drop_duplicates(["Season", "Round"])["TrackTemperature"]
    mean, std = temperatures.mean(), temperatures.std()
    if pd.isna(std) or std == 0:
        return train.copy(), {"mean": None if pd.isna(mean) else float(mean), "std": None, "removed": 0}
    keep = train.TrackTemperature.isna() | train.TrackTemperature.between(mean-3*std, mean+3*std)
    return train[keep].copy(), {"mean": float(mean), "std": float(std), "removed": int((~keep).sum())}


def preprocess_data():
    """Return an unfitted transformer; fit imputation/scaling on training only."""
    numbers = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                 keep_empty_features=True)),
                        ("scale", StandardScaler())])
    return ColumnTransformer([
        ("numeric", numbers, NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop")


def train_model(train, model_name="random_forest", target="FinalPosition"):
    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=15,
                                               min_samples_split=5, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "linear_regression": LinearRegression(),
    }
    model = Pipeline([("preprocess", preprocess_data()), ("model", models[model_name])])
    model.fit(train[FEATURES], train[target])
    return model


def evaluate_model(model, data, target="FinalPosition"):
    predictions = model.predict(data[FEATURES])
    actual = data[target].to_numpy()
    per_team = (pd.DataFrame({"Team": data.TeamName.to_numpy(),
                              "AbsoluteError": np.abs(actual-predictions)})
                .groupby("Team").AbsoluteError.agg(["mean", "count"]))
    return {"MAE": float(mean_absolute_error(actual, predictions)),
            "MSE": float(mean_squared_error(actual, predictions)),
            "R2": float(r2_score(actual, predictions)) if len(actual) > 1 else None,
            "n": len(data),
            "TeamMAE": {team: {"MAE": float(row["mean"]), "n": int(row["count"])}
                        for team, row in per_team.iterrows()}}, predictions


def plot_results(model, test, predictions, output_dir, target="FinalPosition"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=["#2563eb"])
    label = "finishing position" if target == "FinalPosition" else "points"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=test[target], y=predictions, ax=ax, s=65, alpha=.8)
    limits = [min(test[target].min(), predictions.min())-1,
              max(test[target].max(), predictions.max())+1]
    ax.plot(limits, limits, "--", color="#64748b", label="Perfect prediction")
    ax.set(xlabel=f"Actual {label}", ylabel=f"Predicted {label}",
           title="Random Forest | later-race test set", xlim=limits, ylim=limits)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pred_vs_actual.png", dpi=180)
    plt.close(fig)
    names = model.named_steps["preprocess"].get_feature_names_out()
    importance = pd.Series(model.named_steps["model"].feature_importances_, index=names)
    importance.to_csv(output_dir / "feature_importances.csv", header=["importance"])
    top = importance.nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([s.replace("numeric__", "").replace("category__", "") for s in top.index], top.values)
    ax.set(xlabel="Impurity-based importance (not causal)", title="Top 10 Random Forest features")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importances.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(predictions-test[target].to_numpy(), bins=15, ax=ax)
    ax.axvline(0, color="#ef4444", linestyle="--")
    ax.set(xlabel=f"Predicted minus actual {label}", title="Test-set residuals")
    fig.tight_layout()
    fig.savefig(output_dir / "residuals.png", dpi=180)
    plt.close(fig)


def run_experiment(df, output_dir, provenance=None, target="FinalPosition"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = clean_dataset(df, target)
    train, val, test = split_by_race(clean)
    train, outliers = filter_training_outliers(train)
    comparisons, models = {}, {}
    for name in ["random_forest", "gradient_boosting", "linear_regression"]:
        models[name] = train_model(train, name, target)
        comparisons[name], _ = evaluate_model(models[name], val, target)
        LOG.info("Validation %s MAE: %.3f", name, comparisons[name]["MAE"])
    # RF is the prespecified primary model. Alternatives are compared on validation.
    model = models["random_forest"]
    metrics, predictions = evaluate_model(model, test, target)
    baseline = (test.GridPosition.fillna(train.GridPosition.median()).to_numpy()
                if target == "FinalPosition" else np.full(len(test), train[target].mean()))
    splits = {name: {"rows": len(part), "rounds": part.Round.unique().tolist(),
                     "teams": sorted(part.TeamName.unique().tolist())}
              for name, part in [("train", train), ("validation", val), ("test", test)]}
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Completed - Phase 1", "target": target, "year": int(clean.Season.iloc[0]),
        "prediction_cutoff": "After qualifying and grid publication; before race start",
        "split_method": "Chronological whole races; approximately 70/20/10",
        "history_policy": "Walk-forward: each race uses only previously completed rounds",
        "model": "random_forest", "random_state": 42,
        "raw_rows": len(df), "finisher_rows": len(clean), "excluded_rows": len(df)-len(clean),
        "splits": splits, "outlier_filter": outliers, "validation": comparisons, "test": metrics,
        "baseline": {"name": "grid_position" if target == "FinalPosition" else "training_mean",
                     "MAE": float(mean_absolute_error(test[target], baseline)),
                     "MSE": float(mean_squared_error(test[target], baseline)),
                     "R2": float(r2_score(test[target], baseline))},
        "missing_fraction": {k: float(v) for k, v in df[NUMERIC].isna().mean().items()},
        "provenance": provenance or [],
        "versions": {p: version(p) for p in ["fastf1", "pandas", "numpy", "scikit-learn"]},
        "limitations": ["Conditional on finishing: DNFs excluded from supervised training/evaluation.",
                        "Single-season, small test set; not evidence of generalization to future seasons.",
                        "Continuous predictions can tie and do not enforce a unique finishing order.",
                        "Qualifying tire and rain observations are proxies, not race strategy or track wetness.",
                        "Live telemetry, tire degradation, DRS and pit-loss physics are future work."],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    df.to_csv(output_dir / "dataset.csv", index=False)
    result = test[["Season", "Round", "RaceName", "Driver", "TeamName", target]].copy()
    result["Prediction"] = predictions
    result["Baseline"] = baseline
    result["AbsoluteError"] = np.abs(predictions-test[target].to_numpy())
    result.to_csv(output_dir / "test_predictions.csv", index=False)
    assignments = pd.concat([part.assign(Split=name) for name, part in
                             [("train", train), ("validation", val), ("test", test)]])
    assignments[["Season", "Round", "DriverId", "Split"]].to_csv(output_dir / "splits.csv", index=False)
    joblib.dump({"model": model, "features": FEATURES, "target": target}, output_dir / "model.joblib")
    plot_results(model, test, predictions, output_dir, target)
    unit = "positions" if target == "FinalPosition" else "points"
    lines = ["Model training complete.",
             f"Test MAE: {metrics['MAE']:.3f} {unit}",
             f"On average, the model predicts within +/-{metrics['MAE']:.2f} {unit} (MAE; not a confidence interval).",
             f"Test MSE: {metrics['MSE']:.3f}", f"Test R2: {metrics['R2']:.3f}",
             f"Baseline MAE: {report['baseline']['MAE']:.3f}", "Per-team test MAE:"]
    lines += [f"  {team}: {values['MAE']:.2f} (n={values['n']})"
              for team, values in metrics["TeamMAE"].items()]
    lines += ["", *report["limitations"]]
    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary)
    print(summary)
    return report


def predict_file(model_path, input_path, output_path):
    """Load only your own trusted model files: joblib uses Python pickle."""
    bundle = joblib.load(model_path)
    frame = pd.read_csv(input_path, dtype={c: str for c in CATEGORICAL})
    missing = set(bundle["features"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input features: {sorted(missing)}")
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame.loc[frame.GridPosition <= 0, "GridPosition"] = np.nan
    frame[CATEGORICAL] = frame[CATEGORICAL].fillna("Unknown")
    frame["Prediction"] = bundle["model"].predict(frame[bundle["features"]])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Download historical data and evaluate models")
    train.add_argument("--year", type=int, default=2025)
    train.add_argument("--rounds", nargs="+", type=int, default=list(range(1, 25)))
    train.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    train.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    train.add_argument("--offline", action="store_true")
    train.add_argument("--dataset", type=Path, help="Reproduce from the exported feature dataset")
    train.add_argument("--target", choices=["FinalPosition", "Points_Scored"], default="FinalPosition")
    predict = commands.add_parser("predict", help="Predict from pre-race features in a CSV")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fastf1.set_log_level("WARNING")
    if args.command == "predict":
        predict_file(args.model, args.input, args.output)
    else:
        if args.dataset:
            df = pd.read_csv(args.dataset, dtype={c: str for c in CATEGORICAL})
            provenance = [{"dataset": args.dataset.name, "source": "exported historical feature table"}]
        else:
            df, provenance = build_dataset(args.year, args.rounds, args.cache_dir, args.offline)
        run_experiment(df, args.output_dir, provenance, args.target)


if __name__ == "__main__":
    try:
        main()
    except (DataLoadError, ValueError, OSError, KeyError) as exc:
        LOG.error("Pipeline stopped: %s", exc)
        raise SystemExit(1) from exc
