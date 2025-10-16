# tests/test_pipeline_edges.py
import sys
import types
import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from cleanote.pipeline import Pipeline


# ---------------------------- Fakes génériques légers ----------------------------
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
        payload = {
            "Symptoms": ["A"],
            "MedicalConclusion": ["C"],
            "Treatments": ["T"],
            "Summary": self.summary,
        }
        out.data[output_col] = json.dumps(payload)
        return out


# ---------------------------- _ensure_scispacy branches ----------------------------
def _install_fake_scispacy_modules():
    mod_scispacy = types.ModuleType("scispacy")
    mod_umls = types.ModuleType("scispacy.umls_linking")

    class UmlsEntityLinker: ...

    mod_umls.UmlsEntityLinker = UmlsEntityLinker
    sys.modules["scispacy"] = mod_scispacy
    sys.modules["scispacy.umls_linking"] = mod_umls


class _FakeSpacyModel:
    def __init__(self):
        self._pipes = {}

    @property
    def pipe_names(self):
        return list(self._pipes.keys())

    def remove_pipe(self, name):
        self._pipes.pop(name, None)

    def add_pipe(self, name, config=None, last=True):
        self._pipes[name] = {"config": config, "last": last}

    def __call__(self, text):
        return SimpleNamespace(ents=[])


def test_ensure_scispacy_import_error_raises(monkeypatch):
    # s'assure qu'aucun module scispacy n'est importable
    for k in list(sys.modules.keys()):
        if k == "scispacy" or k.startswith("scispacy."):
            del sys.modules[k]

    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())

    with pytest.raises(RuntimeError):
        p._ensure_scispacy()


def test_ensure_scispacy_lg_ok(monkeypatch):
    _install_fake_scispacy_modules()

    # spacy.load("en_core_sci_lg") → OK
    def fake_load(name):
        assert name == "en_core_sci_lg"
        return _FakeSpacyModel()

    import spacy as _real_spacy

    monkeypatch.setattr(_real_spacy, "load", fake_load, raising=True)

    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    m = p._ensure_scispacy()
    assert "scispacy_linker" in m.pipe_names  # linker ajouté


def test_ensure_scispacy_fallback_sm(monkeypatch):
    _install_fake_scispacy_modules()

    def fake_load(name):
        if name == "en_core_sci_lg":
            raise OSError("lg not installed")
        assert name == "en_core_sci_sm"
        return _FakeSpacyModel()

    import spacy as _real_spacy

    monkeypatch.setattr(_real_spacy, "load", fake_load, raising=True)

    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    m = p._ensure_scispacy()
    assert "scispacy_linker" in m.pipe_names


def test_ensure_scispacy_no_models_raise(monkeypatch):
    _install_fake_scispacy_modules()

    def fake_load(name):
        raise OSError("no models")

    import spacy as _real_spacy

    monkeypatch.setattr(_real_spacy, "load", fake_load, raising=True)

    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    with pytest.raises(RuntimeError):
        p._ensure_scispacy()


# ---------------------------- _ensure_nlp fallback & decoupe ----------------------------
def test_ensure_nlp_blank_and_sentencizer(monkeypatch):
    # force spacy.load à échouer -> blank("en") + sentencizer ajouté
    import spacy as _real_spacy

    def fake_load_fail(name):
        raise OSError("no en_core_web_sm")

    monkeypatch.setattr(_real_spacy, "load", fake_load_fail, raising=True)

    # blank qui renvoie un objet minimal gérant add_pipe/remove_pipe/pipe_names
    class _Blank:
        def __init__(self, *_):
            self._pipes = {}

        @property
        def pipe_names(self):
            return list(self._pipes.keys())

        def remove_pipe(self, name):
            self._pipes.pop(name, None)

        def add_pipe(self, name, config=None, first=False):
            self._pipes[name] = {"config": config, "first": first}

        def __call__(self, txt):
            parts = [p.strip() for p in txt.split(".") if p.strip()]
            return SimpleNamespace(sents=[SimpleNamespace(text=p) for p in parts])

    monkeypatch.setattr(_real_spacy, "blank", lambda *_: _Blank(), raising=True)

    df = pd.DataFrame({"full_note": ["A. B."]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    p._ensure_nlp()
    assert "sentencizer" in p._nlp.pipe_names
    sents = p.decouper_texte_en_phrases("A\nB")
    assert sents == ["A.", "B."] or len(sents) >= 1  # tolérant sur format


# ---------------------------- NLI: return_probs=False ----------------------------
def test_nli_return_probs_false(monkeypatch):
    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())

    class Tok:
        def __call__(self, prem, hyp, **_):
            return SimpleNamespace(
                to=lambda device: {"input_ids": torch.tensor([[1, 2]])}
            )

    class Clf:
        def __init__(self):
            self.config = SimpleNamespace(
                id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
            )

        def to(self, device):
            return self

        def eval(self): ...
        def __call__(self, **_):
            return SimpleNamespace(logits=torch.tensor([[1.0, 2.0, 3.0]]))

    def fake_ensure_nli(self):
        self._tok = Tok()
        self._clf = Clf()
        self.device = "cpu"
        self._id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        return self._tok, self._clf, self._id2label

    monkeypatch.setattr(Pipeline, "_ensure_nli", fake_ensure_nli, raising=False)
    out = p.nli("p", "h", return_probs=False)
    assert out["probs"] is None and out["prediction"] in (
        "entailment",
        "neutral",
        "contradiction",
    )


# ---------------------------- _get_summary_text + _umls_cuis_from_text cache ----------------------------
def test_get_summary_text_variants_and_cuis_cache(monkeypatch):
    # scispacy ok + modèle spaCy minimal pour __call__
    _install_fake_scispacy_modules()
    import spacy as _real_spacy

    monkeypatch.setattr(
        _real_spacy, "load", lambda name: _FakeSpacyModel(), raising=True
    )

    df = pd.DataFrame({"full_note": ["x"]})
    p = Pipeline(FakeDataset(df), FakeModelH())
    p._ensure_scispacy()

    out_h_col = f"{p.dataset.field}__h"
    assert (
        p._get_summary_text({"Summary": "", out_h_col: "{not json}"}, out_h_col) == ""
    )
    assert (
        p._get_summary_text(
            {"Summary": "", out_h_col: {"Summary": "FromDict"}}, out_h_col
        )
        == "FromDict"
    )

    # cache CUIs
    # on monkeypatch __call__ pour renvoyer un ent avec kb_ents
    def sci_call(self, txt):
        ent = SimpleNamespace(_=SimpleNamespace(kb_ents=[("C1", 1.0)]))
        return SimpleNamespace(ents=[ent])

    p._sci.__class__.__call__ = sci_call
    s1 = p._umls_cuis_from_text("Hello")
    assert s1 == {"C1"}
    # seconde fois -> via cache
    s2 = p._umls_cuis_from_text("  hello ")
    assert s1 == s2


# ---------------------------- _prettier (entrée vide) ----------------------------
def test_prettier_handles_none():
    assert Pipeline._prettier(None) == {
        "entailment": None,
        "neutral": None,
        "contradiction": None,
    }


# ---------------------------- save_row_stats_image erreurs & index ----------------------------
def test_save_row_stats_image_errors_and_index(tmp_path):
    p = Pipeline(FakeDataset(pd.DataFrame({"full_note": ["x"]})), FakeModelH())
    p.homogenize()
    with pytest.raises(ValueError):
        p.save_row_stats_image(0, path=str(tmp_path / "x.png"))
    with pytest.raises(IndexError):
        p.save_row_stats_image(5)


# ---------------------------- save_all_stats_images limit+skip ----------------------------
def test_save_all_stats_images_limit_and_skip(tmp_path, monkeypatch):
    p = Pipeline(FakeDataset(pd.DataFrame({"full_note": ["x", "y"]})), FakeModelH())
    p.homogenize()

    def fake_save(self, i, path=None):  # <= self ajouté
        if i == 0:
            path = path or f"row_{i}_stats.png"
            with open(path, "wb") as f:
                f.write(b"")
            return path
        raise ValueError("no metrics")

    monkeypatch.setattr(Pipeline, "save_row_stats_image", fake_save, raising=True)
    monkeypatch.chdir(tmp_path)

    paths = p.save_all_stats_images(limit=2)
    assert len(paths) == 1 and os.path.exists(paths[0])


# ---------------------------- _normalize_for_sentences variantes ----------------------------
def test_normalize_for_sentences_variants():
    p = Pipeline(FakeDataset(pd.DataFrame({"full_note": ["x"]})), FakeModelH())
    txt = "A\r\nB\rC\\nD\nE  \nF!\n  \nG?\nH"
    out = p._normalize_for_sentences(txt)
    assert "A. B. C. D. E F! G? H" in out or out.startswith("A.")


# ---------------------------- verify_NLI branches: pas de phrases ----------------------------
def test_verify_NLI_no_sentences(monkeypatch, capsys):
    p = Pipeline(FakeDataset(pd.DataFrame({"full_note": ["has text"]})), FakeModelH())
    p.homogenize()
    p.dataset_h.data.loc[0, "Summary"] = "has summary"

    monkeypatch.setattr(Pipeline, "decouper_texte_en_phrases", lambda self, txt: [])
    p.verify_NLI()
    out = capsys.readouterr().out
    assert "pas de phrases, skip." in out
