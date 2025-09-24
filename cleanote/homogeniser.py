# cleanote/homogeniser.py
from __future__ import annotations
from typing import List, Protocol, Optional, Callable
from .types import Doc, Context
#from .model_loader import ModelLoader  # pour le futur, on ne l'utilise qu'en stub


class _Step(Protocol):
    def run(self, doc: Doc, ctx: Context) -> Doc: ...


class Homogeniser:
    """Step-based homogeniser (API existante)."""

    def __init__(self, steps: List[_Step]) -> None:
        self.steps = steps

    def run(self, doc: Doc, ctx: Context) -> Doc:
        out = doc
        for s in self.steps:
            out = s.run(out, ctx)
        return out

    # ---------- NOUVEAU : mode batch verbeux ----------
    @classmethod
    def from_docs_and_model(
        cls,
        docs: List[Doc],
        model_name: str,
        verbose: bool = True,
    ) -> "HomogeniserBatch":
        """
        Fabrique un runner batch verbeux qui:
        - ne fait aucun vrai traitement,
        - loggue les étapes (démarrage, 'chargement' modèle, progression, fin),
        - retourne les docs inchangés.
        """
        return HomogeniserBatch(docs=docs, model_name=model_name, verbose=verbose)


class HomogeniserBatch:
    """Batch runner verbeux (stub ML + prompt)."""

    def __init__(self, docs: List[Doc], model_name: str, verbose: bool = True) -> None:
        self.docs = docs
        self.model_name = model_name
        self.verbose = verbose

    def run(self, ctx: Context) -> List[Doc]:
        n = len(self.docs)
        if self.verbose:
            print(f"[Homogeniser] Démarrage de l'homogénéisation | {n} document(s)")
            print(f"[Homogeniser] Chargement du modèle '{self.model_name}' (stub) ...")

        # Stub: on ne charge rien vraiment, on pourrait poser un placeholder dans le contexte
        ctx.artifacts.setdefault("_homogeniser_model_name", self.model_name)

        out_docs: List[Doc] = []
        for i, d in enumerate(self.docs, start=1):
            if self.verbose:
                print(f"[Homogeniser] Traitement du document {i}/{n} (id={d.id})")
            # Aucun vrai traitement pour l’instant → identité
            out_docs.append(d)

        if self.verbose:
            print(f"[Homogeniser] Fin de traitement des documents ({n}/{n})")
        return out_docs
