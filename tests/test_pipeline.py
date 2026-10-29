"""Tests for the daily pipeline, using fake download and clean steps (no network)."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from volrisk.models import GJRParams, GJRStudentTGARCH
from volrisk.models.simulate import simulate_garch
from volrisk.pipeline import PipelineError, run

TRUE = GJRParams(mu=0.0, omega=0.02, alpha=0.02, beta=0.88, nu=6.0, gamma=0.15)


@pytest.fixture(scope="module")
def processed_csv(tmp_path_factory):
    returns, _ = simulate_garch(TRUE, n=1500, seed=5)
    path = tmp_path_factory.mktemp("data") / "spy.csv"
    dates = pd.bdate_range("2020-01-01", periods=len(returns))
    pd.DataFrame({"date": dates, "log_return": returns / 100}).to_csv(path, index=False)
    return path


@pytest.fixture
def vix_csv(tmp_path):
    path = tmp_path / "vixcls.csv"
    pd.DataFrame({"date": ["2025-10-30", "2025-10-31"], "vixcls": [17.5, 18.2]}).to_csv(path, index=False)
    return path


def fake_steps(processed_csv, vix_csv, calls):
    def download_prices(ticker, start, force):
        calls.append(("prices", ticker, force))

    def download_vix(force):
        calls.append(("vix", force))
        return vix_csv

    def clean(ticker):
        calls.append(("clean", ticker))
        return processed_csv

    return {"download_prices": download_prices, "download_vix": download_vix, "clean": clean}


def test_run_writes_results_file(tmp_path, processed_csv, vix_csv):
    calls = []
    out = run(results_dir=tmp_path / "results", **fake_steps(processed_csv, vix_csv, calls))
    result = json.loads(out.read_text())

    assert [c[0] for c in calls] == ["prices", "vix", "clean"]  # steps run in order
    assert out.name == f"spy_{result['as_of']}.json"
    assert result["model"] == "GJR-GARCH(1,1)-t"
    assert result["params"]["mu"] == 0.0
    assert 0 < result["var_95_pct"] < result["var_99_pct"]
    assert result["vix"] == {"date": "2025-10-31", "value": 18.2}


def test_rerun_overwrites_instead_of_duplicating(tmp_path, processed_csv, vix_csv):
    steps = fake_steps(processed_csv, vix_csv, [])
    first = run(results_dir=tmp_path, **steps)
    second = run(results_dir=tmp_path, **steps)
    assert first == second
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_vix_failure_does_not_stop_the_run(tmp_path, processed_csv, vix_csv):
    steps = fake_steps(processed_csv, vix_csv, [])

    def broken_vix(force):
        raise ConnectionError("FRED down")

    steps["download_vix"] = broken_vix
    result = json.loads(run(results_dir=tmp_path, **steps).read_text())
    assert result["vix"] is None
    assert result["var_99_pct"] > 0


def test_price_download_failure_stops_the_run(tmp_path, processed_csv, vix_csv):
    steps = fake_steps(processed_csv, vix_csv, [])

    def broken_prices(ticker, start, force):
        raise ConnectionError("Tiingo down")

    steps["download_prices"] = broken_prices
    with pytest.raises(ConnectionError):
        run(results_dir=tmp_path, **steps)
    assert not list(tmp_path.glob("*.json"))  # no half-finished results


def test_too_little_data_is_refused(tmp_path, vix_csv):
    short = tmp_path / "short.csv"
    pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=50), "log_return": 0.001}).to_csv(short, index=False)
    with pytest.raises(PipelineError):
        run(results_dir=tmp_path, **fake_steps(short, vix_csv, []))


def test_non_converged_fit_is_refused(tmp_path, processed_csv, vix_csv):
    class NeverConverges(GJRStudentTGARCH):
        def fit(self, returns):
            return replace(super().fit(returns), converged=False, message="forced failure")

    with pytest.raises(PipelineError, match="did not converge"):
        run(results_dir=tmp_path, model=NeverConverges(zero_mean=True), **fake_steps(processed_csv, vix_csv, []))


def test_force_flag_is_passed_to_downloads(tmp_path, processed_csv, vix_csv):
    calls = []
    run(results_dir=tmp_path, force=True, **fake_steps(processed_csv, vix_csv, calls))
    assert ("prices", "SPY", True) in calls and ("vix", True) in calls
