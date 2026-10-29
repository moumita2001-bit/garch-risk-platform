"""Tests for the FRED downloader. No network: a fake session stands in for FRED."""

import pandas as pd
import pytest
import requests

from volrisk.data import fred

PAYLOAD = {
    "observations": [
        {"date": "2024-01-02", "value": "13.20"},
        {"date": "2024-01-01", "value": "."},      # holiday: missing
        {"date": "2024-01-03", "value": "14.04"},
    ]
}


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeSession:
    """Returns the queued responses in order; an Exception in the queue is raised."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(fred.time, "sleep", lambda s: None)


def test_parse_drops_missing_values_and_sorts():
    df = fred.parse_observations(PAYLOAD, "VIXCLS")
    assert list(df.columns) == ["date", "vixcls"]
    assert df["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df["vixcls"].tolist() == [13.20, 14.04]


def test_parse_rejects_unexpected_response():
    with pytest.raises(fred.FredError):
        fred.parse_observations({"error": "bad"}, "VIXCLS")


def test_retries_after_server_error_then_succeeds():
    session = FakeSession(FakeResponse(503), FakeResponse(200, PAYLOAD))
    df = fred.fetch_series("VIXCLS", "secret-key", session=session)
    assert len(df) == 2
    assert len(session.calls) == 2


def test_retries_after_network_error():
    session = FakeSession(requests.ConnectionError("reset"), FakeResponse(200, PAYLOAD))
    assert len(fred.fetch_series("VIXCLS", "secret-key", session=session)) == 2


def test_bad_request_fails_immediately_without_retry():
    session = FakeSession(FakeResponse(400), FakeResponse(200, PAYLOAD))
    with pytest.raises(fred.FredError):
        fred.fetch_series("VIXCLS", "secret-key", session=session)
    assert len(session.calls) == 1


def test_gives_up_after_all_retries():
    session = FakeSession(*[FakeResponse(503)] * 4)
    with pytest.raises(fred.FredError):
        fred.fetch_series("VIXCLS", "secret-key", session=session, retries=3)


def test_api_key_never_appears_in_errors():
    session = FakeSession(FakeResponse(400))
    with pytest.raises(fred.FredError) as info:
        fred.fetch_series("VIXCLS", "secret-key", session=session)
    assert "secret-key" not in str(info.value)


def test_download_writes_csv_and_log_then_skips_same_day(tmp_path, monkeypatch):
    monkeypatch.setattr(fred, "get_api_key", lambda: "secret-key")
    path = fred.download_series(raw_dir=tmp_path, session=FakeSession(FakeResponse(200, PAYLOAD)))
    assert pd.read_csv(path)["vixcls"].tolist() == [13.20, 14.04]
    log = pd.read_csv(tmp_path / "fred_log.csv")
    assert log["series_id"].tolist() == ["VIXCLS"]
    assert "secret-key" not in (tmp_path / "fred_log.csv").read_text()

    # Second call on the same day: no request is made (the empty session would fail)
    fred.download_series(raw_dir=tmp_path, session=FakeSession())
    assert len(pd.read_csv(tmp_path / "fred_log.csv")) == 1
