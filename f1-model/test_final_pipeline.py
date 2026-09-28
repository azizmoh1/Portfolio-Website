"""Final-stage tests use synthetic fixtures only; no network or invented results."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import final_pipeline as final
import f1_pipeline as base
from test_pipeline import sample_data


def fixture():
    data = sample_data(5)
    data["FieldSize"] = 20
    data["DriverRecentPosition"] = data.GridPosition
    data["QualifyingGapPercent"] = data.GridPosition / 10
    data["QualifyingPosition"] = data.GridPosition
    data.loc[data.DriverId.isin(["driver19", "driver20"]), "Status"] = "Retired"
    data.loc[data.Status.eq("Retired"), "FinalPosition"] = data.loc[data.Status.eq("Retired"), "GridPosition"]
    return data


def test_retirement_targets_exclude_dns_and_disqualification():
    data = fixture()
    data.loc[0, "Status"] = "Disqualified"
    data.loc[1, "Status"] = "Did not start"
    data.loc[2, "Status"] = "Lapped"
    data.loc[3, "Status"] = "Withdrew"
    eligible, labels = final.retirement_data(data)
    assert 0 not in eligible.index and 1 not in eligible.index and 3 not in eligible.index
    assert labels.loc[2] == 0
    assert labels.sum() == 10


def test_unique_complete_field_ranking_and_no_outcome_leakage():
    data = fixture()
    bundle = final.fit_bundle(data, "random_forest", "random_forest")
    race = data[data.Round.eq(5)]
    predicted = final.rank_field(race, bundle)
    assert sorted(predicted.PredictedPosition) == list(range(1, 21))
    assert predicted.RetirementProbability.between(0, 1).all()
    changed = race.assign(FinalPosition=999, Points_Scored=999, Status="Did not start", Finished=False)
    other = final.rank_field(changed, bundle)
    np.testing.assert_array_equal(predicted.PredictedPosition, other.PredictedPosition)
    np.testing.assert_allclose(predicted.RetirementProbability, other.RetirementProbability, atol=1e-12)
    assert not {"FinalPosition", "Status", "Finished", "Points_Scored"} & set(final.FEATURES)


def test_partial_grid_and_duplicate_driver_are_rejected():
    data = fixture()
    bundle = final.fit_bundle(data, "linear_regression", "logistic_regression")
    race = data[data.Round.eq(5)]
    with pytest.raises(ValueError, match="complete field"):
        final.rank_field(race.iloc[:-1], bundle)
    with pytest.raises(ValueError, match="duplicate"):
        final.rank_field(pd.concat([race, race.iloc[:1]]), bundle)


def test_new_drivers_and_teams_do_not_break_prediction():
    data = fixture()
    bundle = final.fit_bundle(data, "gradient_boosting", "logistic_regression")
    race = data[data.Round.eq(5)].copy()
    race["TeamId"] = "new_team"
    race["DriverId"] = [f"rookie_{i}" for i in range(len(race))]
    assert final.rank_field(race, bundle).ExpectedPositionScore.notna().all()


def test_prepare_never_loads_current_race_results(tmp_path, monkeypatch):
    results = pd.DataFrame({"DriverId": ["a", "b"], "Abbreviation": ["AAA", "BBB"],
        "TeamId": ["t1", "t2"], "TeamName": ["One", "Two"], "Position": [1., 2.]})
    qual = SimpleNamespace(results=results, weather_data=pd.DataFrame(),
        laps=pd.DataFrame({"Driver": ["AAA", "BBB"], "Compound": ["SOFT", "SOFT"],
                           "LapTime": pd.to_timedelta([80, 81], unit="s")}),
        event={"EventName": "Future race"})
    calls = []
    def loader(year, number, kind, cache, offline):
        calls.append((year, number, kind))
        assert kind == "Q", "Future-race prepare must never read race outcomes"
        return qual
    monkeypatch.setattr(base, "load_and_cache_session", loader)
    path = tmp_path/"grid.csv"
    pd.DataFrame({"DriverId": ["a", "b"], "GridPosition": [2, 1]}).to_csv(path, index=False)
    frame = final.prepare_grid(2026, 1, path, tmp_path)
    assert calls == [(2026, 1, "Q")]
    assert frame.GridPosition.tolist() == [2, 1]
    assert frame.QualifyingPosition.tolist() == [1., 2.]
    assert frame.QualifyingGapPercent.tolist() == pytest.approx([0., 1.25])
    assert "FinalPosition" not in frame and "Status" not in frame


def test_prepare_earlier_history_cutoff(tmp_path, monkeypatch):
    results = pd.DataFrame({"DriverId": ["a", "b"], "Abbreviation": ["AAA", "BBB"],
        "TeamId": ["t1", "t2"], "TeamName": ["One", "Two"], "Position": [1., 2.]})
    qual = SimpleNamespace(results=results, weather_data=pd.DataFrame(), laps=pd.DataFrame(), event={"EventName": "Future"})
    monkeypatch.setattr(base, "load_and_cache_session", lambda *a, **k: qual)
    def history(year, rounds, cache, offline):
        assert year == 2026 and rounds == [1, 2]
        return fixture(), []
    monkeypatch.setattr(base, "build_dataset", history)
    path = tmp_path/"grid.csv"
    pd.DataFrame({"DriverId": ["a", "b"], "GridPosition": [1, 2]}).to_csv(path, index=False)
    assert len(final.prepare_grid(2026, 3, path, tmp_path)) == 2


def test_constant_baseline_probability_metrics_are_finite():
    metrics = final.probability_metrics([0, 0, 1, 0], [.2]*4)
    assert metrics["Brier"] == pytest.approx(.19)
    assert metrics["ROC_AUC"] == .5


def test_data_loader_rejects_missing_seasons(tmp_path):
    with pytest.raises(ValueError, match="collect first"):
        final.load_data(tmp_path)


def test_populated_but_incomplete_result_response_is_rejected():
    partial = pd.DataFrame({"DriverId": [np.nan, np.nan], "TeamId": [np.nan, np.nan],
                            "Position": [np.nan, np.nan], "Status": [np.nan, np.nan]})
    with pytest.raises(base.DataLoadError, match="Incomplete"):
        base.validate_race_results(partial)
