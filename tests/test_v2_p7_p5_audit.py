from scripts.v2_p7.p5.audit import evaluate_status


def _es(passes=None):
    flags = passes or {}
    return {key: {"pass": flags.get(key, True)} for key in ("ES1", "ES2", "ES3", "ES4", "ES5")}


def test_p5_status_conjoins_all_es_gates():
    result = evaluate_status({"pass": True}, {"pass": True, "recomputed_es3_pass": True}, _es())
    assert result["status"] == "pass"


def test_p5_status_fails_when_es_gate_fails():
    result = evaluate_status(
        {"pass": True},
        {"pass": True, "recomputed_es3_pass": True},
        _es({"ES1": False}),
    )
    assert result["status"] == "fail"
    assert result["es_pass"] is False


def test_p5_status_fails_when_es3_recompute_disagrees():
    result = evaluate_status(
        {"pass": True},
        {"pass": True, "recomputed_es3_pass": False},
        _es({"ES3": True}),
    )
    assert result["status"] == "fail"
    assert result["es3_consistent"] is False
