"""Deterministic tests: fixtures are synthetic and never used as portfolio results."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import f1_pipeline as f1


def sample_data(n_races=10):
    rows = []
    for race in range(1, n_races+1):
        for driver in range(1, 21):
            rows.append({"Season": 2025, "Round": race, "RaceName": f"Race {race}",
                         "DriverId": f"driver{driver}", "Driver": f"D{driver}",
                         "TeamId": f"t{(driver-1)//2}", "TeamName": f"Team {(driver-1)//2}",
                         "GridPosition": driver, "DriverExperience": race-1,
                         "TeamReliability": np.nan if race == 1 else 95.,
                         "CarPerformanceIndex": np.nan if race == 1 else float(driver),
                         "TrackTemperature": 30.+race, "AirTemperature": 23.,
                         "TireCompound": "3", "TrackStatus": "Dry",
                         "FinalPosition": (driver+race-2) % 20 + 1,
                         "Points_Scored": max(0, 11-driver),
                         "Status": "Finished", "Finished": True})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("status,expected", [("Finished", True), ("+1 Lap", True),
    ("+2 Laps", True), ("Lapped", True), ("Engine", False), ("Retired", False),
    ("Did not start", False), ("Disqualified", False), ("Not classified", False)])
def test_finisher_status(status, expected):
    assert f1.finished(status) is expected


def test_split_never_mixes_races_or_reverses_time():
    data = sample_data(24)
    train, val, test = f1.split_by_race(data.sample(frac=1, random_state=1))
    assert [part.Round.nunique() for part in [train, val, test]] == [16, 5, 3]
    assert train.Round.max() < val.Round.min() <= val.Round.max() < test.Round.min()
    assert set(train.index) | set(val.index) | set(test.index) == set(data.index)
    assert set(train.index).isdisjoint(test.index)
    assert len(f1.split_by_race(sample_data(3))[2]) == 20
    with pytest.raises(ValueError, match="three races"):
        f1.split_by_race(sample_data(2))


def test_history_uses_last_five_races_not_five_cars_and_excludes_future():
    history = sample_data(8)
    history.loc[history.Round.eq(1), "FinalPosition"] = 20
    history.loc[history.Round.between(2, 6), "FinalPosition"] = 4
    # Current and future results must not affect the features for round 7.
    history.loc[history.Round.ge(7), "FinalPosition"] = 1
    result = pd.DataFrame([{"DriverId": "driver1", "Abbreviation": "D1",
                            "TeamId": "t0", "TeamName": "Team 0", "GridPosition": 3,
                            "Position": 1, "Points": 25, "Status": "Finished"}])
    row = f1.extract_features(result, None, history, 2025, 7, "Test").iloc[0]
    assert row.CarPerformanceIndex == 4
    assert row.DriverExperience == 6
    assert row.TeamReliability == 100
    row = f1.extract_features(result, None, history, 2026, 1, "Test").iloc[0]
    assert row.DriverExperience == 0
    assert pd.isna(row.TeamReliability)


def test_weather_ffill_is_within_session_and_unknown_is_not_dry():
    weather = pd.DataFrame({"Time": [3, 1, 2], "TrackTemp": [np.nan, 30., 36.],
                            "AirTemp": [22., 20., np.nan], "Rainfall": [True, False, False]})
    track, air, condition = f1.weather_features(weather)
    assert track == 34
    assert air == pytest.approx(62/3)
    assert condition == "Mixed Rain"
    assert f1.weather_features(pd.DataFrame())[2] == "Unknown"


def test_compound_comes_from_fastest_valid_qualifying_lap():
    session = SimpleNamespace(weather_data=pd.DataFrame(), laps=pd.DataFrame({
        "Driver": ["D1", "D1", "D1"], "LapTime": [60, 61, 62],
        "Compound": ["SOFT", "MEDIUM", "HARD"], "Deleted": [True, False, False]}))
    _, compounds = f1.qualifying_features(session)
    assert compounds["D1"] == "2"


def test_cleaning_excludes_dnfs_and_handles_pitlane_grid():
    data = sample_data()
    data.loc[0, "Finished"] = False
    data.loc[1, "GridPosition"] = 0
    cleaned = f1.clean_dataset(data)
    assert len(cleaned) == len(data)-1
    assert cleaned.loc[cleaned.DriverId.eq("driver2") & cleaned.Round.eq(1), "GridPosition"].isna().all()
    with pytest.raises(ValueError, match="Duplicate"):
        f1.clean_dataset(pd.concat([data, data.iloc[:1]]))


def test_imputation_unseen_teams_and_no_target_leakage():
    train, val, _ = f1.split_by_race(sample_data())
    train["TrackTemperature"] = np.nan
    train["CarPerformanceIndex"] = np.nan
    val["TeamName"] = "Unseen Team"
    for name in ["random_forest", "gradient_boosting", "linear_regression"]:
        model = f1.train_model(train, name)
        prediction = model.predict(val[f1.FEATURES])
        assert np.isfinite(prediction).all()
        altered = val.assign(FinalPosition=1000, Points_Scored=1000)
        np.testing.assert_allclose(prediction, model.predict(altered), rtol=1e-12, atol=1e-12)
        assert not any("FinalPosition" in feature or "Points_Scored" in feature
                       for feature in model["preprocess"].get_feature_names_out())


def test_training_outlier_filter_keeps_missing_and_removes_extreme():
    data = sample_data(20)
    data["TrackTemperature"] = 30.
    data.loc[data.Round.eq(20), "TrackTemperature"] = 1000
    data.loc[data.Round.eq(1), "TrackTemperature"] = np.nan
    clean, stats = f1.filter_training_outliers(data)
    assert stats["removed"] == 20
    assert 20 not in clean.Round.unique()
    assert 1 in clean.Round.unique()


def test_offline_failure_is_clear_and_does_not_call_network(tmp_path, monkeypatch):
    monkeypatch.setattr(f1.requests, "get", lambda *a, **k: pytest.fail("Network in offline mode"))
    with pytest.raises(f1.DataLoadError, match="No cached"):
        f1.fallback_results(2025, 1, tmp_path, offline=True)


def test_full_experiment_and_saved_model_inference(tmp_path):
    data = sample_data()
    report = f1.run_experiment(data, tmp_path)
    assert report["test"]["n"] == 20
    for name in ["metrics.json", "dataset.csv", "test_predictions.csv", "splits.csv",
                 "model.joblib", "pred_vs_actual.png", "feature_importances.png", "residuals.png"]:
        assert (tmp_path/name).stat().st_size > 0
    data.tail(20)[f1.FEATURES].to_csv(tmp_path/"input.csv", index=False)
    f1.predict_file(tmp_path/"model.joblib", tmp_path/"input.csv", tmp_path/"prediction.csv")
    assert pd.read_csv(tmp_path/"prediction.csv").Prediction.notna().all()


def test_points_target_has_no_position_feature(tmp_path):
    train, _, test = f1.split_by_race(sample_data())
    model = f1.train_model(train, target="Points_Scored")
    metrics, predictions = f1.evaluate_model(model, test, "Points_Scored")
    assert metrics["n"] == len(test)
    assert np.isfinite(predictions).all()
