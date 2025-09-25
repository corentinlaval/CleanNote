# tests/test_pipeline.py
import pandas as pd

# Adapte ces imports selon ton arborescence de projet :
# from ton_module.pipeline import Pipeline
# Ici on suppose pipeline.py à la racine:
from cleanote.pipeline import Pipeline


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


def test_pipeline_apply_happy_path(capsys):
    ds = FakeDataset(field="full_note")
    m_h = FakeModel()
    m_v = FakeModel()  # inutilisé dans la version actuelle du pipeline

    pipe = Pipeline(dataset=ds, model_h=m_h, model_v=m_v)
    out = pipe.apply()

    # Vérifie les prints
    printed = capsys.readouterr().out
    assert "[Pipeline] Starting pipeline..." in printed
    assert "[Pipeline] Start Homogenization..." in printed
    assert "[Pipeline] Homogenization completed." in printed
    assert "[Pipeline] Pipeline completed." in printed

    # Vérifie que model_h.run a été appelé avec le bon prompt + output_col
    assert len(m_h.calls) == 1
    call = m_h.calls[0]
    assert call["prompt"].startswith("please give me the number of words")
    assert call["output_col"] == f"{ds.field}__h"

    # Vérifie le DataFrame de sortie: colonne ajoutée, contenu cohérent
    assert out is not ds  # retourne un *nouveau* dataset
    assert out.field == ds.field  # le champ est préservé
    new_col = f"{ds.field}__h"
    assert new_col in out.data.columns
    assert list(out.data[new_col]) == ["OK-0", "OK-1"]

    # L'original n'est pas modifié
    assert new_col not in ds.data.columns


def test_pipeline_no_side_effect_on_input():
    ds = FakeDataset()
    m_h = FakeModel()
    pipe = Pipeline(ds, m_h, FakeModel())
    _ = pipe.apply()

    # Le dataset d'entrée ne doit pas être modifié
    assert list(ds.data.columns) == ["index", ds.field]
