from cleanote.types import Doc, Context, Issue
from cleanote.pipeline import Pipeline
from cleanote.homogeniser import Homogeniser
from cleanote.verifier import Verifier
from cleanote.model_loader import ModelLoader


# downloader factice
class DummyDownloader:
    def __init__(self): ...
    def fetch(self, ctx):
        yield Doc(id="d1", text="hello   world")
        yield Doc(id="d2", text="")


# micro-étapes & checks minimalistes
class NormalizeWhitespace:
    def __init__(self): ...
    def run(self, doc, ctx):
        import re

        return doc.copy(update={"text": re.sub(r"\s+", " ", doc.text).strip()})


def check_not_empty(doc, ctx):
    if doc.text:
        return []
    return [Issue(code="empty", message="text is empty", severity="warn")]

    # def test_pipeline_end_to_end():
    pipe = Pipeline(
        downloader=DummyDownloader(),
        homogeniser=Homogeniser([NormalizeWhitespace()]),
        verifier=Verifier([check_not_empty]),
        models=ModelLoader("dummy"),
    )
    docs, reports = pipe.run(Context(run_id="t"))
    assert [d.id for d in docs] == ["d1", "d2"]
    assert docs[0].text == "hello world"
    assert any(i.code == "empty" for i in reports[-1].issues)
