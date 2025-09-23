from .types import Context


class ModelLoader:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._cache: dict[str, object] = {}

    def preload(self, ctx: Context) -> None:
        """Charge les artefacts en mémoire (stub pour l’instant)."""
        # Exemple plus tard: self._cache["rules"] = load_rules(self.model_name)
        ctx.artifacts.update(self._cache)

    def get(self, key: str):
        return self._cache.get(key)
