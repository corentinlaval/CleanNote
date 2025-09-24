# tests/test_homogeniser.py
import pytest
from cleanote.types import Doc, Context
from cleanote.homogeniser import Homogeniser
from cleanote.model_loader import ModelLoader


def test_homogeniser_with_single_doc(capsys):
    ctx = Context(run_id="test")
    doc = Doc(id="1", text="hello")

    ml = ModelLoader("dummy")
    hom = Homogeniser()

    out = hom.run(ml, doc, ctx)

    # ✅ output
    assert isinstance(out, Doc)
    assert out.text == "hello"

    # ✅ logs
    captured = capsys.readouterr().out
    assert "[Homogeniser] Initializing model loader" in captured
    assert "Sending 1 document(s)" in captured
    assert "[Homogeniser] Done." in captured


def test_homogeniser_with_multiple_docs(capsys):
    ctx = Context(run_id="test")
    docs = [Doc(id="1", text="foo"), Doc(id="2", text="bar")]

    ml = ModelLoader("dummy")
    hom = Homogeniser()

    out = hom.run(ml, docs, ctx)

    # output
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(d, Doc) for d in out)

    # logs
    captured = capsys.readouterr().out
    assert "Sending 2 document(s)" in captured
    assert "Model loader returned 2 document(s)" in captured
