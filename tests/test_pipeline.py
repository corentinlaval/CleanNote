# tests/test_pipeline_more_edges.py
import builtins
import sys
from types import SimpleNamespace
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from cleanote.pipeline import Pipeline


# --------- petits fakes utilitaires réutilisables ---------
class FakeDataset:
    def __init__(self, df: pd.DataFrame, field: str = "full_note"):
        self.data = df
        self.field = field


class FakeModelH:
    def __init__(self, prompt=None, summary="S"):
        self.prompt = prompt
        self.summary = summary

    def run(self, dataset, output_col="full_note__h", **_):
        out = FakeDataset(dataset.data.copy(), dataset.field)
        out.data[output_col] = json.dumps(
            {
                "Symptoms": ["A"],
                "MedicalConclusion": ["C"],
                "Treatments": ["T"],
                "Summary": self.summary,
            }
        )
        return out


# ---------------------------- _ensure_scispacy branches ----------------------------


def test_ensure_scispacy_fallback_sm_and_add_pipe(monkeypatch):
    """
    Couvre :
      - import 'scispacy' OK
      - spacy.load(model_lg) -> OSError, fallback sur 'en_core_sci_sm'
      - remove_pipe('scispacy_linker') si déjà présent
      - add_pipe('scispacy_linker', config={...})
    """

    # dummy modules pour "import scispacy" et "from scispacy.umls_linking import UmlsEntityLinker"
    class _Umls:
        class UmlsEntityLinker: ...

    dummy_scispacy = SimpleNamespace(umls_linking=_Umls)
    monkeypatch.setitem(sys.modules, "scispacy", dummy_scispacy)
    monkeypatch.setitem(sys.modules, "scispacy.umls_linking", _Umls)

    # fake NLP objet
    class NLP:
        def __init__(self):
            self._pipes = {"scispacy_linker": {}}

        @property
        def pipe_names(self):
            return list(self._pipes.keys())

        def remove_pipe(self, name):
            self._pipes.pop(name, None)

        def add_pipe(self, name, config=None, last=True):
            self._pipes[name] = {"config": config, "last": last}

    # spacy.load : 1er appel -> OSError (lg), 2e -> NLP() (sm)
    calls = {"i": 0}

    def fake_spacy_load(name):
        calls["i"] += 1
        if calls["i"] == 1:
            raise OSError("lg not installed")
        return NLP()

    import cleanote.pipeline as pl

    monkeypatch.setattr(
        pl, "spacy", SimpleNamespace(load=fake_spacy_load), raising=False
    )

    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    nlp = p._ensure_scispacy()

    # add_pipe bien présent avec une config seuil (float) issue de UMLS_MIN_SCORE
    assert "scispacy_linker" in nlp.pipe_names
    cfg = nlp._pipes["scispacy_linker"]["config"]
    assert isinstance(cfg.get("threshold"), float)


def test_ensure_scispacy_import_error_raises(monkeypatch):
    """import scispacy échoue -> RuntimeError explicite."""
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "scispacy":
            raise ImportError("no module")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    with pytest.raises(RuntimeError) as e:
        p._ensure_scispacy()
    assert "SciSpaCy n'est pas installé" in str(e.value)


def test_ensure_scispacy_no_model_raises(monkeypatch):
    """lg -> OSError, sm -> OSError : on remonte une RuntimeError 'Aucun modèle SciSpaCy'."""

    # dummy modules scispacy
    class _U:
        class UmlsEntityLinker: ...

    monkeypatch.setitem(sys.modules, "scispacy", SimpleNamespace(umls_linking=_U))
    monkeypatch.setitem(sys.modules, "scispacy.umls_linking", _U)

    def fake_spacy_load_always_fail(name):
        raise OSError("nope")

    import cleanote.pipeline as pl

    monkeypatch.setattr(
        pl, "spacy", SimpleNamespace(load=fake_spacy_load_always_fail), raising=False
    )

    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    with pytest.raises(RuntimeError) as e:
        p._ensure_scispacy()
    assert "Aucun modèle SciSpaCy" in str(e.value)


# ---------------------------- _ensure_nlp fallback ----------------------------


def test_ensure_nlp_blank_fallback_and_sentencizer(monkeypatch):
    """spacy.load('en_core_web_sm') -> OSError => spacy.blank('en') + ajout sentencizer."""

    class DummyNLP:
        def __init__(self):
            self._pipes = {}

        @property
        def pipe_names(self):
            return list(self._pipes.keys())

        def remove_pipe(self, name):
            self._pipes.pop(name, None)

        def add_pipe(self, name, **kw):
            self._pipes[name] = kw

    class _FakeSpacy:
        def load(self, name):
            raise OSError("no sm")

        def blank(self, lang):
            return DummyNLP()

    import cleanote.pipeline as pl

    monkeypatch.setattr(pl, "spacy", _FakeSpacy(), raising=False)

    p = Pipeline(FakeDataset(pd.DataFrame({"full_note": ["x"]})), FakeModelH())
    p._ensure_nlp()
    assert "sentencizer" in p._nlp.pipe_names


# ---------------------------- _ensure_nli idempotence + nli(probs=False) ----------------------------


def test_ensure_nli_idempotence_and_nli_no_probs(monkeypatch):
    class Tok:
        def __call__(self, prem, hyp, **kw):
            return SimpleNamespace(
                to=lambda device: {"input_ids": torch.tensor([[1, 2, 3]])}
            )

    class Clf:
        def __init__(self):
            self.config = SimpleNamespace(
                id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
            )

        def to(self, device):
            return self

        def eval(self): ...
        def __call__(self, **inp):
            return SimpleNamespace(logits=torch.tensor([[2.0, 1.0, 0.5]]))

    import cleanote.pipeline as pl

    monkeypatch.setattr(
        pl,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_: Tok()),
        raising=False,
    )
    monkeypatch.setattr(
        pl,
        "AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=lambda *_: Clf()),
        raising=False,
    )
    monkeypatch.setattr(pl, "torch", torch, raising=False)

    p = Pipeline(FakeDataset(pd.DataFrame({"full_note": ["x"]})), FakeModelH())
    t1 = p._ensure_nli()
    t2 = p._ensure_nli()
    assert t1 == t2  # idempotent

    out = p.nli("P", "H", return_probs=False)
    assert out["prediction"] in ("entailment", "neutral", "contradiction")
    assert out["probs"] is None


# ---------------------------- _row_metrics_dict + _prettier + clipping image ----------------------------


def test_row_metrics_dict_and_prettier_and_image_clipping(tmp_path, monkeypatch):
    # dataset + homogenize minimal
    df = pd.DataFrame({"full_note": ["note0", "note1"]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    p.homogenize()

    # crée des colonnes avec valeurs en dehors de [0,1] pour tester clipping dans l'image
    ddf = p.dataset_h.data
    ddf.loc[0, "nli_ent_mean"] = 1.5
    ddf.loc[0, "nli_neu_mean"] = -0.2
    ddf.loc[0, "nli_con_mean"] = 0.4
    ddf.loc[0, "umls_match_rate"] = 2.0
    ddf.loc[0, "umls_loss_rate"] = -1.0
    ddf.loc[0, "umls_creation_rate"] = 0.5
    ddf.loc[0, "umls_jaccard"] = 0.3
    # colonnes Symptoms* absentes -> _row_metrics_dict renvoie None pour ces clés

    # _row_metrics_dict : valeurs présentes / None
    md = p._row_metrics_dict(ddf.loc[0])
    assert md["NLI – entailment"] == 1.5
    assert md["Symptoms – match rate"] is None

    # _prettier complète les clés manquantes
    assert set(Pipeline._prettier({"probs": {}}).keys()) == {
        "entailment",
        "neutral",
        "contradiction",
    }
    assert all(v is None for v in Pipeline._prettier({"probs": {}}).values())

    # génération image : vérifie qu’un fichier est bien produit (et donc clipping appliqué sans crasher)
    out = p.save_row_stats_image(0, path=str(tmp_path / "row0.png"))
    assert os.path.exists(out)


# ---------------------------- _umls_match_bulk avec seuil élevé ----------------------------


def test_umls_match_bulk_threshold(monkeypatch):
    # scispacy présent
    class _U:
        class UmlsEntityLinker: ...

    monkeypatch.setitem(sys.modules, "scispacy", SimpleNamespace(umls_linking=_U))
    monkeypatch.setitem(sys.modules, "scispacy.umls_linking", _U)

    # NLP renvoie des entités avec score 0.5
    class NLP:
        @property
        def pipe_names(self):
            return []

        def remove_pipe(self, *_): ...
        def add_pipe(self, *a, **k): ...
        def pipe(self, texts, **_):
            class _Ent:
                def __init__(self, kb):
                    self._ = SimpleNamespace(kb_ents=kb)

            class _Doc:
                def __init__(self, ents):
                    self.ents = ents

            return [_Doc([_Ent([("CUI", 0.5)])])] * len(texts)

    def fake_spacy_load(name):
        return NLP()

    import cleanote.pipeline as pl

    monkeypatch.setattr(
        pl, "spacy", SimpleNamespace(load=fake_spacy_load), raising=False
    )

    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    p.UMLS_MIN_SCORE = 0.9  # seuil haut -> aucun match
    p.homogenize()

    out = p._umls_match_bulk(["term1", "term2"])
    assert out == {"term1": False, "term2": False}
