import pytest

from scripts.v2_p5b.family_txn_spike import FamilyCampaign, InjectedCrash, run_spike


def test_partial_promotion_aborts_and_retries(tmp_path):
    c = FamilyCampaign(tmp_path / "campaign")
    with pytest.raises(InjectedCrash):
        c.attempt("20260717", crash_after=1)
    out = c.recover()
    assert out["status"] == "ABORTED"
    assert c.ledger == []
    c.retry("20260717")
    assert len(c.ledger) == 1
    assert c.read_visible()["generation"] == 1


def test_committed_recovery_does_not_duplicate_ledger(tmp_path):
    c = FamilyCampaign(tmp_path / "campaign")
    c.attempt("20260717")
    c.recover()
    assert len(c.ledger) == 1
    c.recover()
    assert len(c.ledger) == 1


def test_crash_during_abort_is_idempotent(tmp_path):
    c = FamilyCampaign(tmp_path / "campaign")
    with pytest.raises(InjectedCrash):
        c.attempt("20260717", crash_after=2)
    with pytest.raises(InjectedCrash):
        c.recover(crash_during_recovery=True)
    assert c.journal["status"] == "ABORTING"
    c.recover()
    assert c.journal["status"] == "ABORTED"
    c.retry("20260717")
    assert len(c.ledger) == 1


def test_ten_day_spike_has_one_entry_per_day(tmp_path):
    result = run_spike(tmp_path / "campaign", days=10)
    assert result["status"] == "PASS"
    assert result["ledger_entries"] == 10
    assert result["generation"] == 10
    assert len(result["crashes"]) == 3
    assert result["visible"]["generation"] == 10
