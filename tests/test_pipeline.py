# test_homogeniser.py
# Pytest tests for Homogeniser.run
# Adjust the import below to match your package/module structure.
from dataclasses import dataclass
from typing import List

import pytest

# >>> Adjust this import <<<
# from yourpkg.homogeniser import Homogeniser
# If Homogeniser is in the same folder as this test, uncomment:
# from homogeniser import Homogeniser

# ------- Minimal stand-ins for your project types -------


@dataclass
class Doc:
    id: str
    content: str


@dataclass
class Context:
    user: str = "test-user"


class ModelLoader:
    """Interface hint. Concrete fake below must implement initialize() and transform()."""

    def initialize(self) -> None: ...
    def transform(self, doc: Doc, ctx: Context) -> Doc: ...


# ------- Fake model loader for testing -------


class FakeModelLoader(ModelLoader):
    def __init__(self, name: str = "FakeML"):
        self.name = name
        self.initialized = False
        self.transform_calls: List[Doc] = []

    def initialize(self) -> None:
        if self.initialized:
            raise AssertionError("initialize() called more than once")
        self.initialized = True

    def transform(self, doc: Doc, ctx: Context) -> Doc:
        self.transform_calls.append(doc)
        # Example transformation: append a marker + user to content (to check propagation of ctx)
        return Doc(
            id=doc.id,
            content=f"{doc.content} | processed-by:{self.name} | ctx:{ctx.user}",
        )


# ------- Tests -------


def test_run_single_doc(monkeypatch, capsys):
    # Import here to avoid import path issues during collection if you adjust paths
    from yourpkg.homogeniser import Homogeniser  # <<< change to your real module

    ml = FakeModelLoader(name="UnitModel")
    h = (
        Homogeniser(steps=[])
        if "steps" in Homogeniser.__init__.__code__.co_varnames
        else Homogeniser()
    )  # compat

    doc_in = Doc(id="d1", content="Hello")
    ctx = Context(user="u1")

    out = h.run(ml, doc_in, ctx)

    # Type/shape assertions
    assert isinstance(out, Doc), "Expected a single Doc when input is a single Doc"
    assert out.id == "d1"
    assert "processed-by:UnitModel" in out.content
    assert "ctx:u1" in out.content

    # Behavior assertions
    assert ml.initialized is True, "Model loader should be initialized exactly once"
    assert (
        len(ml.transform_calls) == 1
    ), "transform() should be called once for single input"

    # Log assertions
    captured = capsys.readouterr().out
    assert "Initializing model loader 'UnitModel'" in captured
    assert "Initialization completed." in captured
    assert "Sending 1 document(s) to the model loader" in captured
    assert "Model loader returned 1 document(s)." in captured
    assert "Done." in captured


def test_run_multiple_docs_preserves_order(monkeypatch, capsys):
    from yourpkg.homogeniser import Homogeniser  # <<< change to your real module

    ml = FakeModelLoader(name="BatchModel")
    h = (
        Homogeniser(steps=[])
        if "steps" in Homogeniser.__init__.__code__.co_varnames
        else Homogeniser()
    )  # compat

    docs_in = [
        Doc(id="a", content="First"),
        Doc(id="b", content="Second"),
        Doc(id="c", content="Third"),
    ]
    ctx = Context(user="u2")

    out_list = h.run(ml, docs_in, ctx)

    # Type/shape assertions
    assert isinstance(out_list, list), "Expected a list when input is a list"
    assert [d.id for d in out_list] == [
        "a",
        "b",
        "c",
    ], "Output order must match input order"

    # Each doc transformed
    for out_doc, in_doc in zip(out_list, docs_in):
        assert in_doc.id == out_doc.id
        assert f"processed-by:BatchModel" in out_doc.content
        assert "ctx:u2" in out_doc.content

    # Behavior assertions
    assert ml.initialized is True
    assert len(ml.transform_calls) == len(
        docs_in
    ), "transform() should be called once per input doc"
    assert [d.id for d in ml.transform_calls] == ["a", "b", "c"]

    # Logs
    captured = capsys.readouterr().out
    assert "Initializing model loader 'BatchModel'" in captured
    assert "Sending 3 document(s) to the model loader" in captured
    assert "Model loader returned 3 document(s)." in captured
    assert "Done." in captured


def test_run_empty_list(monkeypatch, capsys):
    """Optional edge case: if an empty list is passed, ensure it returns an empty list and logs correctly."""
    from yourpkg.homogeniser import Homogeniser  # <<< change to your real module

    ml = FakeModelLoader(name="EdgeModel")
    h = (
        Homogeniser(steps=[])
        if "steps" in Homogeniser.__init__.__code__.co_varnames
        else Homogeniser()
    )  # compat

    out_list = h.run(ml, [], Context())

    assert isinstance(out_list, list)
    assert out_list == [], "Should return an empty list on empty input"
    assert ml.initialized is True
    assert len(ml.transform_calls) == 0

    captured = capsys.readouterr().out
    assert "Sending 0 document(s) to the model loader" in captured
    assert "Model loader returned 0 document(s)." in captured
    assert "Done." in captured
