# tests/test_pipeline.py
import os
import re
import pandas as pd

from cleanote.pipeline import Pipeline


# ----------------- Doubles de test -----------------
class FakeDataset:
    """Dataset minimal compatible (DataFrame-only) pour tester Pipeline."""

    def __init__(self, field="full_note"):
        self.field = field
        self.name = "dummy/ds"
        self.limit = 2
        self.data = pd.DataFrame(
            {"index": [0, 1], field: ["hello world", "second row"]}
        )


class FakeModel:
    """Mock de Model avec .run(dataset, prompt, output_col)."""

    def __init__(self):
        self.calls = []  # trace des appels

    def run(self, dataset, prompt, output_col=None):
        # On trace les arguments pour les assertions
        self.calls.append({"prompt": prompt, "output_col": output_col})
        # Retourne une *copie légère* du dataset avec la nouvelle colonne
        out = type(dataset).__new__(type(dataset))
        out.__dict__ = dict(dataset.__dict__)
        df = dataset.data.copy()
        df[output_col] = ["OK-0", "OK-1"]
        out.data = df
        return out


# ----------------- Tests -----------------
def test_pipeline_apply_happy_path(capsys):
    ds = FakeDataset(field="full_note")
    m_h = FakeModel()

    pipe = Pipeline(dataset=ds, model_h=m_h)
    out = pipe.apply()

    # Vérifie les prints
    printed = capsys.readouterr().out
    assert "[Pipeline] Starting pipeline..." in printed
    assert "[Pipeline] Prompt for Homogenization:" in printed
    assert "[Pipeline] Start Homogenization..." in printed
    assert "[Pipeline] Homogenization completed." in printed
    assert "[Pipeline] Pipeline completed." in printed

    # run() a bien été appelé une fois avec le bon prompt et la bonne colonne
    assert len(m_h.calls) == 1
    call = m_h.calls[0]
    assert call["prompt"].lstrip().startswith("Analyze the document below")
    assert call["output_col"] == f"{ds.field}__h"

    # Vérifie l'objet retourné : nouveau dataset avec la nouvelle colonne
    assert out is not ds
    assert out.field == ds.field
    new_col = f"{ds.field}__h"
    assert new_col in out.data.columns
    assert list(out.data[new_col]) == ["OK-0", "OK-1"]

    # L'original n'est pas modifié
    assert new_col not in ds.data.columns


def test_pipeline_no_side_effect_on_input():
    ds = FakeDataset()
    m_h = FakeModel()
    _ = Pipeline(ds, m_h).apply()

    # Le dataset d'entrée ne doit pas être modifié
    assert list(ds.data.columns) == ["index", ds.field]


def test_build_prompt_contains_required_keys():
    """Le prompt doit mentionner explicitement les 4 clés JSON attendues et 'valid JSON'."""
    ds = FakeDataset()
    m_h = FakeModel()
    p = Pipeline(ds, m_h)
    prompt = p.build_prompt_h()

    for key in ["Symptoms", "MedicalConclusion", "Treatments", "Summary"]:
        assert key in prompt

    # Le prompt doit rappeler 'valid JSON'
    assert re.search(r"\bvalid JSON\b", prompt, re.IGNORECASE) is not None

    # Le prompt doit contenir 'Document:'
    assert "Document:" in prompt


def test_output_col_name_depends_on_field():
    """Le suffixe __h doit s'appuyer sur dataset.field."""
    ds = FakeDataset(field="note_text")
    m_h = FakeModel()
    _ = Pipeline(ds, m_h).apply()

    assert len(m_h.calls) == 1
    assert m_h.calls[0]["output_col"] == "note_text__h"


def test_homogenize_returns_none_and_sets_dataset_h():
    """homogenize() ne retourne rien mais doit positionner self.dataset_h."""
    ds = FakeDataset()
    m_h = FakeModel()
    p = Pipeline(ds, m_h)

    ret = p.homogenize()
    assert ret is None
    assert p.dataset_h is not None
    # colonne créée
    assert f"{ds.field}__h" in p.dataset_h.data.columns


def test_to_excel_success(tmp_path, monkeypatch):
    """to_excel crée un fichier et renvoie son chemin (par défaut 'dataset_h.xlsx')."""
    ds = FakeDataset()
    m_h = FakeModel()
    p = Pipeline(ds, m_h)
    _ = p.apply()

    # Monkeypatch: évite la dépendance à openpyxl dans le CI
    def fake_to_excel(self, path, index=False):
        p = os.fspath(path)
        with open(p, "wb") as f:
            f.write(b"OK")

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        path = p.to_excel()
        assert path == "dataset_h.xlsx"
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.chdir(cwd)
