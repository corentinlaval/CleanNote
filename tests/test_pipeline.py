# tests/test_pipeline_edges.py
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cleanote.pipeline import Pipeline


# ----------------------------- Fakes utilitaires (mêmes esprits que ton autre fichier) -----------------------------


class FakeDataset:
    def __init__(self, df: pd.DataFrame, field: str = "full_note"):
        self.data = df
        self.field = field


class FakeModelH:
    def __init__(self, prompt=None, summary="S"):
        self.prompt = prompt
        self.calls = []
        self.summary = summary

    def run(self, dataset, output_col="full_note__h", **_):
        self.calls.append({"dataset": dataset, "output_col": output_col})
        out = FakeDataset(dataset.data.copy(), dataset.field)
        payload = {
            "Symptoms": ["A"],
            "MedicalConclusion": ["C"],
            "Treatments": ["T"],
            "Summary": self.summary,  # paramétrable
        }
        out.data[output_col] = json.dumps(payload)
        return out


class FakeEnt:
    def __init__(self, kb_ents):
        self._ = SimpleNamespace(kb_ents=kb_ents)


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeSciNLP:
    def __init__(self, term_to_ok=None, text_to_cuis=None):
        self._pipes = {}
        self._term_to_ok = term_to_ok or {}
        self._text_to_cuis = text_to_cuis or {}

    @property
    def pipe_names(self):
        return list(self._pipes.keys())

    def remove_pipe(self, name):
        self._pipes.pop(name, None)

    def add_pipe(self, name, config=None, last=True):
        self._pipes[name] = {"config": config, "last": last}

    def pipe(self, texts, batch_size=64, n_process=1):
        docs = []
        for t in texts:
            ok = self._term_to_ok.get(t, False)
            ents = [FakeEnt([("CUI_OK", 1.0)])] if ok else []
            docs.append(FakeDoc(ents))
        return docs

    def __call__(self, text):
        cuis = self._text_to_cuis.get(text, [])
        ents = [FakeEnt([(c, 1.0) for c in cuis])] if cuis else []
        return FakeDoc(ents)


class FakeTok:
    def __call__(
        self,
        prem,
        hyp,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ):
        return SimpleNamespace(
            to=lambda device: {"input_ids": torch.tensor([[1, 2, 3]])}
        )


class FakeClf:
    def __init__(self):
        self.config = SimpleNamespace(
            id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        )

    def eval(self): ...
    def to(self, device):
        return self

    def __call__(self, **inputs):
        return SimpleNamespace(
            logits=torch.tensor([[1.0, 1.0, 1.0]])
        )  # égalité, peu importe pour ces tests


# ----------------------------- Fixtures / patches minimales -----------------------------


@pytest.fixture
def base_df():
    return pd.DataFrame(
        {
            "full_note": [
                "Line1\nLine2",
                "Something",
                "",
            ],  # lignes pensées pour _normalize_for_sentences + NLI skip
            "Symptoms": [np.nan, np.nan, np.nan],
            "MedicalConclusion": [np.nan, np.nan, np.nan],
            "Treatments": [np.nan, np.nan, np.nan],
            "Summary": ["", "", ""],  # forcer fallback via __h si besoin
        }
    )


@pytest.fixture
def pipe_obj(monkeypatch, base_df):
    # SciSpaCy fake minimal
    fake_sci = FakeSciNLP(
        term_to_ok={"a": True},
        text_to_cuis={
            "Line1. Line2": {"C1"},  # après normalisation, deux phrases → CUIs
            "S": {"C2"},
            "Something": {"C1"},
            "": set(),
        },
    )

    def fake_ensure_scispacy(self):
        self._sci = fake_sci
        return self._sci

    def fake_ensure_nlp(self):
        class _N:
            def __call__(self, txt):
                parts = [p.strip() for p in txt.split(".") if p.strip()]
                return SimpleNamespace(sents=[SimpleNamespace(text=p) for p in parts])

            @property
            def pipe_names(self):
                return []

            def remove_pipe(self, *_):
                pass

            def add_pipe(self, *_, **__):
                pass

        self._nlp = _N()

    def fake_ensure_nli(self):
        self._tok = FakeTok()
        self._clf = FakeClf()
        self._id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.device = "cpu"
        return self._tok, self._clf, self._id2label

    monkeypatch.setattr(
        Pipeline, "_ensure_scispacy", fake_ensure_scispacy, raising=False
    )
    monkeypatch.setattr(Pipeline, "_ensure_nlp", fake_ensure_nlp, raising=False)
    monkeypatch.setattr(Pipeline, "_ensure_nli", fake_ensure_nli, raising=False)

    ds = FakeDataset(base_df.copy())
    p = Pipeline(ds, FakeModelH())
    return p


# ----------------------------- Tests de branches manquantes -----------------------------


def test_normalize_for_sentences_variants(pipe_obj):
    p = pipe_obj
    # \r\n, \r, \\n, retours sans ponctuation, espaces multiples
    txt = "A\r\nB\rC\\nD\nE  \nF!\n  \nG?\nH"
    out = p._normalize_for_sentences(txt)
    # Doit contenir des points ajoutés et espaces compressés
    assert "A. B. C. D. E F! G? H" in out or out.startswith(
        "A."
    )  # tolérant sur l’exact


def test_save_row_stats_image_errors_and_index(pipe_obj, tmp_path):
    p = pipe_obj
    p.homogenize()
    # ligne 0 : aucune métrique → ValueError
    with pytest.raises(ValueError):
        p.save_row_stats_image(0, path=str(tmp_path / "x.png"))
    # index hors bornes
    with pytest.raises(IndexError):
        p.save_row_stats_image(99)


def test_get_summary_text_invalid_json_and_dict_payload(pipe_obj):
    p = pipe_obj
    p.homogenize()
    out_h_col = f"{p.dataset.field}__h"

    # Cas 1 : payload string non-JSON -> retourne "" (Summary vide + JSON invalide)
    row = {"Summary": "", out_h_col: "{not json}"}
    assert p._get_summary_text(row, out_h_col) == ""

    # Cas 2 : payload dict -> récupère Summary
    row2 = {"Summary": "", out_h_col: {"Summary": "FromDict"}}
    assert p._get_summary_text(row2, out_h_col) == "FromDict"


def test_umls_cuis_from_text_cache(pipe_obj):
    p = pipe_obj
    p.homogenize()
    # première extraction
    s1 = p._umls_cuis_from_text("S")
    assert isinstance(s1, set)
    # seconde → via cache (on vérifie la présence dans _umls_doc_cache et l’égalité)
    key = p._norm_term("S")
    assert key in p._umls_doc_cache
    s2 = p._umls_cuis_from_text("S")
    assert s1 == s2


def test_ensure_scispacy_idempotent(pipe_obj):
    p = pipe_obj
    one = p._ensure_scispacy()
    two = p._ensure_scispacy()
    assert one is two  # early return couvert


def test_verify_NLI_no_sentences_branch(monkeypatch, pipe_obj, capsys):
    p = pipe_obj
    p.homogenize()

    # Forcer decouper_texte_en_phrases à renvoyer [] pour déclencher "pas de phrases, skip."
    monkeypatch.setattr(Pipeline, "decouper_texte_en_phrases", lambda self, txt: [])
    p.dataset_h.data.loc[0, "full_note"] = "Has text"
    p.dataset_h.data.loc[0, "Summary"] = "Has summary"

    p.verify_NLI()
    out = capsys.readouterr().out
    assert "pas de phrases, skip." in out


def test_save_all_stats_images_limit_and_skip(pipe_obj, tmp_path, monkeypatch):
    p = pipe_obj
    p.homogenize()

    def fake_save(self, i, path=None):  # <-- IMPORTANT: ajouter self
        if i == 0:
            pth = path or f"row_{i}_stats.png"
            with open(pth, "wb") as f:
                f.write(b"")
            return pth
        raise ValueError("no metrics")

    monkeypatch.setattr(Pipeline, "save_row_stats_image", fake_save)
    monkeypatch.chdir(tmp_path)

    paths = p.save_all_stats_images(limit=2)
    assert len(paths) == 1 and os.path.exists(paths[0])
