from cleanote.types import Doc, Context, Issue
from cleanote.verifier import Verifier


def check_min_length(n: int):
    def _check(doc, ctx):
        return (
            []
            if len(doc.text) >= n
            else [Issue(code="min_length", message="too short", severity="warn")]
        )

    return _check


def test_verifier_returns_issue_without_modifying_text():
    v = Verifier([check_min_length(5)])
    doc = Doc(id="n1", text="a")
    d2, rep = v.run(doc, Context(run_id="t", params={}))
    assert d2.text == doc.text  # non destructif
    assert any(i.code == "min_length" for i in rep.issues)
