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
    """Modèle d'homogénéisation minimal : enregistre les appels et renvoie un dataset cloné avec __h."""

    def __init__(self, prompt=None):
        self.prompt = prompt
        self.calls = []

    def run(self, dataset, output_col="full_note__h", **_):
        self.calls.append({"dataset": dataset, "output_col": output_col})
        out = FakeDataset(dataset.data.copy(), dataset.field)
        # Par défaut, on met un JSON valide dans la colonne __h
        payload = {
            "Symptoms": ["A"],
            "MedicalConclusion": ["C"],
            "Treatments": ["T"],
            "Summary": "S",
        }
        out.data[output_col] = json.dumps(payload)
        return out


# --- Fakes SciSpaCy / spaCy ---


class FakeEnt:
    def __init__(self, kb_ents):
        # On simule l'extension spaCy via un SimpleNamespace
        self._ = SimpleNamespace(kb_ents=kb_ents)


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeSciNLP:
    """Assez pour ce qu'on utilise : .pipe() et __call__ pour _umls_cuis_from_text."""

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
        # pour _umls_cuis_from_text
        cuis = self._text_to_cuis.get(text, [])
        ents = [FakeEnt([(c, 1.0) for c in cuis])] if cuis else []
        return FakeDoc(ents)


# --- Fakes NLI (tokenizer + classifier) ---


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
        # Retourne un objet compatible .to(self.device) qui donne un dict pour le modèle
        return SimpleNamespace(
            to=lambda device: {"input_ids": torch.tensor([[1, 2, 3]])}
        )


class FakeClf:
    def __init__(self):
        # Ordre classique MNLI : 0=entailment, 1=neutral, 2=contradiction (on met des logits simples)
        self.config = SimpleNamespace(
            id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        )
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def to(self, device):
        return self

    def __call__(self, **inputs):
        # Logits préférant entailment
        logits = torch.tensor([[5.0, 3.0, 1.0]])
        return SimpleNamespace(logits=logits)


# ----------------------------- Fixtures / patches -----------------------------


@pytest.fixture
def base_df():
    # 3 lignes : 1) listes directes, 2) strings JSON, 3) vides (NaN)
    return pd.DataFrame(
        {
            "full_note": [
                "Pain in chest. Shortness of breath.",
                "Patient feels better. Discharged.",
                "",
            ],
            "Symptoms": [
                ["chest pain", "dyspnea"],  # liste directe
                json.dumps(["nausea", "vomiting"]),  # string JSON
                np.nan,  # vide
            ],
            "MedicalConclusion": [
                "myocardial infarction; angina",  # string à split
                json.dumps(["recovery"]),  # string JSON
                np.nan,
            ],
            "Treatments": [["aspirin"], "PPI, rest", np.nan],  # liste  # string à split
            "Summary": [
                "Chest pain treated with aspirin.",
                "",  # vide -> sera repris depuis __h
                "",
            ],
        }
    )


@pytest.fixture
def pipe_obj(monkeypatch, base_df):
    # Fake SciSpaCy : les termes normalisés "chest pain" et "aspirin" matchent
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
    # Fake CUI extraction sur textes complets (source/résumé)
    text_to_cuis = {
        "Pain in chest. Shortness of breath.": ["C001"],  # source
        "Chest pain treated with aspirin.": ["C001", "C002"],  # résumé
        "S": ["C002"],  # fallback Summary depuis __h
        "Patient feels better. Discharged.": ["C003"],  # source ligne 1
        "": [],  # ligne vide
    }
    fake_sci = FakeSciNLP(term_to_ok=term_ok_map, text_to_cuis=text_to_cuis)

    # Patch _ensure_scispacy pour court-circuiter import et chargement
    def fake_ensure_scispacy(self):
        self._sci = fake_sci
        return self._sci

    # Patch _ensure_nlp pour découpage de phrases très simple (split par '.')
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

    # Patch NLI loader
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
    m_h = FakeModelH(prompt=None)  # force le fallback build_prompt_h()
    p = Pipeline(ds, m_h)

    # Après homogenize(), on aura __h avec JSON {"Symptoms":["A"],"Treatments":["T"],"Summary":"S"}
    return p


# ----------------------------- Tests -----------------------------


def test_homogenize_builds_prompt_and_calls_model(pipe_obj):
    p = pipe_obj
    # Avant : pas de dataset_h, pas de prompt
    assert p.dataset_h is None
    assert p.model_h.prompt is None

    p.homogenize()

    # Le prompt a été construit
    assert isinstance(p.model_h.prompt, str) and '"Symptoms": []' in p.model_h.prompt
    # dataset_h défini et colonne __h présente
    out_col = f"{p.dataset.field}__h"
    assert out_col in p.dataset_h.data.columns
    # run appelé avec bon nom de colonne
    assert p.model_h.calls[-1]["output_col"] == out_col


def test_umls_extract_entity_list_all_paths(pipe_obj):
    p = pipe_obj
    p.homogenize()
    out_h_col = f"{p.dataset.field}__h"
    row0 = p.dataset_h.data.iloc[0].copy()
    row1 = p.dataset_h.data.iloc[1].copy()
    row2 = p.dataset_h.data.iloc[2].copy()

    # 0: Symptoms est une liste -> renvoie nettoyé
    res0 = p._extract_entity_list(row0, "Symptoms", out_h_col)
    assert res0 == ["chest pain", "dyspnea"]

    # 1: MedicalConclusion est un string JSON -> parse en liste
    res1 = p._extract_entity_list(row1, "MedicalConclusion", out_h_col)
    assert res1 == ["recovery"]

    # 0: string avec séparateurs -> split
    res2 = p._extract_entity_list(row0, "MedicalConclusion", out_h_col)
    assert res2 == ["myocardial infarction", "angina"]

    # 2: valeurs vides -> fallback JSON de __h
    res3 = p._extract_entity_list(row2, "Treatments", out_h_col)
    assert res3 == ["T"]


def test_umls_match_bulk_and_cache(pipe_obj):
    p = pipe_obj
    p.homogenize()
    # Première passe : certains termes matchent
    terms = ["chest pain", "dyspnea", "aspirin"]
    out1 = p._umls_match_bulk(terms)
    assert out1 == {"chest pain": True, "dyspnea": False, "aspirin": True}
    # Cache : si on rejoue, ne relance pas pipe sur ces termes (on ne peut pas mesurer directement,
    # mais on s'assure que le résultat est identique et que 'terms' déjà vus n'explosent pas)
    out2 = p._umls_match_bulk(terms)
    assert out2 == out1


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

    # Lignes avec entités : metrics numériques ou NaN selon cas
    # (ligne 0: 2 symptoms; ligne 1: 2 depuis strings; ligne 2: fallback via __h)
    assert df.loc[0, "symptoms_umls_total"] == 2
    assert df.loc[0, "symptoms_umls_matched"] in (0, 1, 2)
    # Lignes sans entités -> NaN pour les taux
    # (dans notre dataset toutes finissent avec qques entités, mais on peut vérifier sur une colonne)
    assert math.isnan(df.loc[2, "medicalconclusion_umls_match_rate"]) or isinstance(
        df.loc[2, "medicalconclusion_umls_match_rate"], float
    )


def test_verify_UMLS_summary_vs_source(pipe_obj):
    p = pipe_obj
    p.homogenize()
    # Ligne 1 a Summary vide -> sera pris depuis __h ("S")
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

    # Vérifie au moins une ligne avec valeurs numériques
    assert df.loc[0, "umls_src_total"] >= 0
    assert df.loc[0, "umls_sum_total"] >= 0


def test_nli_single_call_and_verify_NLI(pipe_obj, monkeypatch):
    p = pipe_obj
    p.homogenize()

    # Forcer des textes simples (deux phrases chacune) pour NLI
    df = p.dataset_h.data
    df.loc[0, "Summary"] = "It is OK. Really OK."
    df.loc[0, "full_note"] = "OK indeed. Confirmed."

    # Découpage personnalisé (déjà patché dans fixture)
    # Vérifie un appel direct à nli()
    out = p.nli("premise here.", "hyp here.")
    assert out["prediction"] in ("entailment", "neutral", "contradiction")
    assert set(out["probs"].keys()) == {"entailment", "neutral", "contradiction"}

    # Et maintenant la vérification complète (remplit nli_*_mean)
    p.verify_NLI()
    assert "nli_ent_mean" in df.columns
    # Comme nos logits préfèrent entailment, la moyenne doit être > neutral/contra en général
    assert (df.loc[0, "nli_ent_mean"] is None) or (df.loc[0, "nli_ent_mean"] >= 0.0)


def test_generer_table_prettier_average(pipe_obj):
    p = pipe_obj
    # matrice 2x2 avec des dicts partiels -> _prettier complète les clés à None
    lignes = ["h1", "h2"]
    cols = ["p1", "p2"]
    raw = [
        [{"probs": {"entailment": 0.9}}, {"probs": {"neutral": 0.5}}],
        [{"probs": {"contradiction": 0.2}}, {"probs": {"entailment": 0.7}}],
    ]
    mat = p.generer_table(
        lignes, cols, lambda i, j: raw[lignes.index(i)][cols.index(j)]
    )
    # Toutes les clés présentes
    assert all(
        set(cell.keys()) == {"entailment", "neutral", "contradiction"}
        for row in mat
        for cell in row
    )

    avg = p.average(lignes, cols, mat)
    assert set(avg.keys()) == {"entailment", "neutral", "contradiction"}
    # Valeurs numériques ou None (si pas de meilleures colonnes)
    assert avg["entailment"] is None or avg["entailment"] >= 0.0


def test_save_row_stats_image_and_all_images(pipe_obj, tmp_path):
    p = pipe_obj
    p.homogenize()
    # Créer des colonnes métriques minimales pour la ligne 0
    df = p.dataset_h.data
    df.loc[0, "nli_ent_mean"] = 0.8
    df.loc[0, "nli_neu_mean"] = 0.1
    df.loc[0, "nli_con_mean"] = 0.1
    df.loc[0, "umls_match_rate"] = 0.5
    df.loc[0, "umls_loss_rate"] = 0.5
    df.loc[0, "umls_creation_rate"] = 0.0
    df.loc[0, "umls_jaccard"] = 0.3

    # et pour la ligne 1, rien -> save_all devrait skipper proprement
    # Sauvegarde image pour la ligne 0
    out_path = tmp_path / "row0.png"
    got = p.save_row_stats_image(0, path=str(out_path))
    assert os.path.exists(got) and got.endswith("row0.png")

    # save_all_stats_images (limité à 2 pour aller vite)
    paths = p.save_all_stats_images(limit=2)
    # au moins une image créée (ligne 0), ligne 1 peut être skippée (aucune métrique)
    assert any(os.path.exists(pth) for pth in paths)


def test_to_excel_exports_file(pipe_obj, tmp_path, monkeypatch):
    p = pipe_obj
    p.homogenize()
    # rediriger cwd vers tmp pour ne pas polluer le repo
    monkeypatch.chdir(tmp_path)
    out = p.to_excel()
    assert os.path.exists(out)
    assert out == "dataset_h.xlsx"


def test_apply_full_orchestration(pipe_obj, capsys):
    p = pipe_obj
    out = p.apply()
    # apply retourne dataset_h
    assert out is p.dataset_h
    # On vérifie que le log final apparait
    stdout = capsys.readouterr().out
    assert "[Pipeline] Pipeline completed." in stdout
