import pytest

from src.data import _retry, coverage_gaps


def test_coverage_complete():
    assert coverage_gaps(2024, "2024-03-20", "2024-09-30", ("2024-03-20", "2024-09-30")) == []


def test_coverage_missing_tail_and_head():
    gaps = coverage_gaps(2022, "2022-04-08", "2022-09-30", ("2022-04-07", "2022-10-05"))
    assert len(gaps) == 2
    assert any("ends 2022-09-30" in g for g in gaps)


def test_retry_succeeds_after_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert _retry(flaky, tries=3, wait_s=0) == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_exhaustion():
    def always():
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        _retry(always, tries=2, wait_s=0)
