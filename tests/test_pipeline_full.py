# tests/test_pipeline_full.py
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cleanote.pipeline import Pipeline


# ----------------------------- Fakes utilitaires -----------------------------
class FakeDataset:
    def __init__(self, df: pd.DataFrame, field: str = "full_note"):
        self.data = df
        self.field = field


class FakeModelH:
    def __init__(self, prompt=None):
        self.prompt = prompt
        self.calls = []

    def run(self, dataset, output_col="full_note__h", **_):
        self.calls.append({"dataset": dataset, "output_col": output_col})
        out = FakeDataset(dataset.data.copy(), dataset.field)
        payload = {
            "Symptoms": ["A"],
            "MedicalConclusion": ["C"],
            "Treatments": ["T"],
            "Summary": "S",
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
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def to(self, device):
        return self

    def __call__(self, **inputs):
        logits = torch.tensor([[5.0, 3.0, 1.0]])
        return SimpleNamespace(logits=logits)


# ----------------------------- Fixtures / patches -----------------------------
@pytest.fixture
def base_df():
    return pd.DataFrame(
        {
            "full_note": [
                "Pain in chest. Shortness of breath.",
                "Patient feels better. Discharged.",
                "",
            ],
            "Symptoms": [
                ["chest pain", "dyspnea"],
                json.dumps(["nausea", "vomiting"]),
                np.nan,
            ],
            "MedicalConclusion": [
                "myocardial infarction; angina",
                json.dumps(["recovery"]),
                np.nan,
            ],
            "Treatments": [["aspirin"], "PPI, rest", np.nan],
            "Summary": [
                "Chest pain treated with aspirin.",
                "",
                "",
            ],
        }
    )


@pytest.fixture
def pipe_obj(monkeypatch, base_df):
    term_ok_map = {
        "chest pain": True,
        "dyspnea": False,
        "nausea": True,
        "vomiting": False,
        "myocardial infarction": True,
        "angina": False,
        "aspirin": True,
        "ppi": False,
        "rest": False,
    }
    text_to_cuis = {
        "Pain in chest. Shortness of breath.": ["C001"],
        "Chest pain treated with aspirin.": ["C001", "C002"],
        "S": ["C002"],
        "Patient feels better. Discharged.": ["C003"],
        "": [],
    }
    fake_sci = FakeSciNLP(term_to_ok=term_ok_map, text_to_cuis=text_to_cuis)

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
        self.device = "cpu"
        self._id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        return self._tok, self._clf, self._id2label

    monkeypatch.setattr(
        Pipeline, "_ensure_scispacy", fake_ensure_scispacy, raising=False
    )
    monkeypatch.setattr(Pipeline, "_ensure_nlp", fake_ensure_nlp, raising=False)
    monkeypatch.setattr(Pipeline, "_ensure_nli", fake_ensure_nli, raising=False)

    ds = FakeDataset(base_df.copy())
    m_h = FakeModelH(prompt=None)
    p = Pipeline(ds, m_h)
    return p


# ----------------------------- Tests -----------------------------
def test_homogenize_builds_prompt_and_calls_model(pipe_obj):
    p = pipe_obj
    assert p.dataset_h is None and p.model_h.prompt is None
    p.homogenize()
    assert isinstance(p.model_h.prompt, str) and '"Symptoms": []' in p.model_h.prompt
    out_col = f"{p.dataset.field}__h"
    assert out_col in p.dataset_h.data.columns
    assert p.model_h.calls[-1]["output_col"] == out_col


def test_umls_extract_entity_list_all_paths(pipe_obj):
    p = pipe_obj
    p.homogenize()
    out_h_col = f"{p.dataset.field}__h"
    row0 = p.dataset_h.data.iloc[0].copy()
    row1 = p.dataset_h.data.iloc[1].copy()
    row2 = p.dataset_h.data.iloc[2].copy()

    assert p._extract_entity_list(row0, "Symptoms", out_h_col) == [
        "chest pain",
        "dyspnea",
    ]
    assert p._extract_entity_list(row1, "MedicalConclusion", out_h_col) == ["recovery"]
    assert p._extract_entity_list(row0, "MedicalConclusion", out_h_col) == [
        "myocardial infarction",
        "angina",
    ]
    assert p._extract_entity_list(row2, "Treatments", out_h_col) == ["T"]


def test_umls_match_bulk_and_cache(pipe_obj):
    p = pipe_obj
    p.homogenize()
    terms = ["chest pain", "dyspnea", "aspirin"]
    out1 = p._umls_match_bulk(terms)
    assert out1 == {"chest pain": True, "dyspnea": False, "aspirin": True}
    assert p._umls_match_bulk(terms) == out1  # cache idem


def test_verify_QuickUMLS_creates_and_fills_columns(pipe_obj):
    p = pipe_obj
    p.homogenize()
    p.verify_QuickUMLS()
    df = p.dataset_h.data
    for short in ("symptoms", "medicalconclusion", "treatments"):
        for suffix in (
            "umls_total",
            "umls_matched",
            "umls_match_rate",
            "umls_loss_rate",
        ):
            assert f"{short}_{suffix}" in df.columns
    assert df.loc[0, "symptoms_umls_total"] == 2
    assert df.loc[0, "symptoms_umls_matched"] in (0, 1, 2)
    assert math.isnan(df.loc[2, "medicalconclusion_umls_match_rate"]) or isinstance(
        df.loc[2, "medicalconclusion_umls_match_rate"], float
    )


def test_verify_UMLS_summary_vs_source(pipe_obj):
    p = pipe_obj
    p.homogenize()
    p.verify_UMLS_summary_vs_source()
    df = p.dataset_h.data
    for col in [
        "umls_src_total",
        "umls_sum_total",
        "umls_overlap_count",
        "umls_match_rate",
        "umls_loss_rate",
        "umls_creation_rate",
        "umls_jaccard",
    ]:
        assert col in df.columns
    assert df.loc[0, "umls_src_total"] >= 0
    assert df.loc[0, "umls_sum_total"] >= 0


def test_nli_single_call_and_verify_NLI(pipe_obj):
    p = pipe_obj
    p.homogenize()
    df = p.dataset_h.data
    df.loc[0, "Summary"] = "It is OK. Really OK."
    df.loc[0, "full_note"] = "OK indeed. Confirmed."
    out = p.nli("premise here.", "hyp here.")
    assert out["prediction"] in ("entailment", "neutral", "contradiction")
    assert set(out["probs"].keys()) == {"entailment", "neutral", "contradiction"}
    p.verify_NLI()
    assert "nli_ent_mean" in df.columns
    assert (df.loc[0, "nli_ent_mean"] is None) or (df.loc[0, "nli_ent_mean"] >= 0.0)


def test_generer_table_prettier_average(pipe_obj):
    p = pipe_obj
    lignes = ["h1", "h2"]
    cols = ["p1", "p2"]
    raw = [
        [
            {"probs": {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}},
            {"probs": {"entailment": 0.4, "neutral": 0.5, "contradiction": 0.1}},
        ],
        [
            {"probs": {"entailment": 0.2, "neutral": 0.2, "contradiction": 0.6}},
            {"probs": {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}},
        ],
    ]
    mat = p.generer_table(
        lignes, cols, lambda i, j: raw[lignes.index(i)][cols.index(j)]
    )
    assert all(
        set(cell.keys()) == {"entailment", "neutral", "contradiction"}
        for row in mat
        for cell in row
    )
    avg = p.average(lignes, cols, mat)
    assert set(avg.keys()) == {"entailment", "neutral", "contradiction"}
    import math as _m

    assert all(isinstance(avg[k], float) and _m.isfinite(avg[k]) for k in avg)


def test_save_row_stats_image_and_all_images(pipe_obj, tmp_path):
    p = pipe_obj
    p.homogenize()
    df = p.dataset_h.data
    df.loc[0, "nli_ent_mean"] = 0.8
    df.loc[0, "nli_neu_mean"] = 0.1
    df.loc[0, "nli_con_mean"] = 0.1
    df.loc[0, "umls_match_rate"] = 0.5
    df.loc[0, "umls_loss_rate"] = 0.5
    df.loc[0, "umls_creation_rate"] = 0.0
    df.loc[0, "umls_jaccard"] = 0.3

    out_path = tmp_path / "row0.png"
    got = p.save_row_stats_image(0, path=str(out_path))
    assert os.path.exists(got) and got.endswith("row0.png")

    paths = p.save_all_stats_images(limit=2)
    assert any(os.path.exists(pth) for pth in paths)


def test_to_excel_exports_file(pipe_obj, tmp_path, monkeypatch):
    p = pipe_obj
    p.homogenize()
    monkeypatch.chdir(tmp_path)

    def fake_to_excel(self, path, index=False):
        with open(path, "wb") as f:
            f.write(b"")

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)
    out = p.to_excel()
    assert out == "dataset_h.xlsx" and os.path.exists(out)
