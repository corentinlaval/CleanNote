from cleanote.types import Context
from cleanote.model_loader import ModelLoader


def test_model_loader_preload_updates_context():
    ml = ModelLoader(model_name="dummy")
    ctx = Context(run_id="t")
    ml.preload(ctx)  # stub pour l’instant
    assert isinstance(ctx.artifacts, dict)


def test_model_loader_attributes_and_getter():
    ml = ModelLoader(model_name="rules-model")

    # model_name doit être conservé
    assert ml.model_name == "rules-model"

    # simulate adding to cache
    ml._cache["foo"] = "bar"

    # get doit retourner la valeur si elle existe
    assert ml.get("foo") == "bar"

    # et None si elle n’existe pas
    assert ml.get("baz") is None
