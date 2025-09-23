from cleanote.types import Doc, Context, Issue
from cleanote.pipeline import Pipeline
from cleanote.homogeniser import Homogeniser
from cleanote.verifier import Verifier
from cleanote.model_loader import ModelLoader


# --- Doubles de test ---------------------------------------------------------


class DummyDownloaderNonEmpty:
    def fetch(self, ctx):
        yield Doc(id="d1", text="hello   world")
        yield Doc(id="d2", text="")


class DummyDownloaderEmpty:
    def fetch(self, ctx):
        if False:  # itérateur vide, jamais exécuté
            yield Doc(id="x", text="")


class NormalizeWhitespace:
    def run(self, doc, ctx):
        import re

        return doc.copy(update={"text": re.sub(r"\s+", " ", doc.text).strip()})


def check_not_empty(doc, ctx):
    if doc.text:
        return []
    return [Issue(code="empty", message="text is empty", severity="warn")]


class SpyModelLoader(ModelLoader):
    def __init__(self):
        super().__init__(model_name="dummy")
        self.called = False

    def preload(self, ctx):
        self.called = True
        super().preload(ctx)


# --- Tests -------------------------------------------------------------------


def test_pipeline_with_models_and_verifier():
    pipe = Pipeline(
        downloader=DummyDownloaderNonEmpty(),
        homogeniser=Homogeniser([NormalizeWhitespace()]),
        verifier=Verifier([check_not_empty]),
        models=SpyModelLoader(),
    )
    ctx = Context(run_id="t")
    docs, reports = pipe.run(ctx)

    # Models.preload() appelé
    assert isinstance(pipe.models, SpyModelLoader)
    assert pipe.models.called is True

    # Docs transformés
    assert [d.id for d in docs] == ["d1", "d2"]
    assert docs[0].text == "hello world"

    # Verifier produit un report (le 2e doc est vide)
    assert len(reports) == 2
    assert any(i.code == "empty" for i in reports[-1].issues)


def test_pipeline_without_verifier_reports_empty():
    pipe = Pipeline(
        downloader=DummyDownloaderNonEmpty(),
        homogeniser=Homogeniser([NormalizeWhitespace()]),
        verifier=None,
        models=None,
    )
    docs, reports = pipe.run(Context(run_id="t"))
    assert [d.id for d in docs] == ["d1", "d2"]
    assert docs[0].text == "hello world"
    assert reports == []  # pas de verifier => pas de rapports


def test_pipeline_empty_downloader_returns_empty_lists():
    pipe = Pipeline(
        downloader=DummyDownloaderEmpty(),
        homogeniser=Homogeniser([NormalizeWhitespace()]),
        verifier=None,
        models=None,
    )
    docs, reports = pipe.run(Context(run_id="t"))
    assert docs == []
    assert reports == []
