from cleanote.types import Context
from cleanote.model_loader import ModelLoader


def test_model_loader_preload_updates_context():
    ml = ModelLoader(model_name="dummy")
    ctx = Context(run_id="t")
    ml.preload(ctx)  # stub pour l’instant
    # On peut au moins vérifier que ça ne casse rien et que artifacts est un dict
    assert isinstance(ctx.artifacts, dict)
