from cleanote.types import Doc, Context, Issue, Report


def test_doc_validation():
    d = Doc(id="n1", text="hello")
    assert d.id == "n1" and d.text == "hello"


def test_report_structure():
    r = Report(doc_id="n1", issues=[Issue(code="x", message="m", severity="warn")])
    assert r.doc_id == "n1"
    assert r.issues[0].severity == "warn"


def test_context_artifacts():
    ctx = Context(run_id="t1")
    ctx.artifacts["rules"] = {"°C": "C"}
    assert "rules" in ctx.artifacts
