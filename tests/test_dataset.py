# tests/test_dataset.py
import pandas as pd
import pytest

# On importe la classe à tester
# Adapte l'import selon ton arborescence :
# from ton_module.dataset import Dataset
from dataset import Dataset


def make_stream_iter(rows):
    """Fabrique un itérateur 'streaming' qui renvoie des dicts comme HF."""

    def _iter():
        for r in rows:
            yield r

    return _iter()


def test_download_happy_path(monkeypatch):
    """
    Cas nominal :
    - streaming=True est bien passé à load_dataset
    - 'field' existe dans les lignes
    - dataframe construit avec le bon nombre de lignes et colonnes
    """
    captured = {}

    def fake_load_dataset(name, split=None, streaming=False, **kwargs):
        # on capture les arguments pour vérification
        captured["name"] = name
        captured["split"] = split
        captured["streaming"] = streaming
        # on renvoie un itérateur de 5 lignes
        rows = [{"text": f"note {i}", "other": i} for i in range(5)]
        return make_stream_iter(rows)

    monkeypatch.setattr("dataset.load_dataset", fake_load_dataset)

    ds = Dataset(name="dummy/repo", split="train", field="text", limit=3)

    # vérifie l'appel à load_dataset
    assert captured["name"] == "dummy/repo"
    assert captured["split"] == "train"
    assert captured["streaming"] is True

    # vérifie le DataFrame
    assert isinstance(ds.data, pd.DataFrame)
    assert len(ds.data) == 3
    assert list(ds.data.columns) == ["index", "text"]
    # contenu
    assert ds.data.loc[0, "index"] == 0
    assert ds.data.loc[0, "text"] == "note 0"
    assert ds.data.loc[2, "text"] == "note 2"


def test_download_missing_field_raises(monkeypatch):
    """
    Champ manquant : déclenche un KeyError dès la 1re ligne.
    """

    def fake_load_dataset(name, split=None, streaming=False, **kwargs):
        rows = [
            {"text": "ok"},  # 'missing' n'existe pas
            {"text": "ok 2"},
        ]
        return make_stream_iter(rows)

    monkeypatch.setattr("dataset.load_dataset", fake_load_dataset)

    ds = Dataset.__new__(Dataset)  # on bypasse __init__ pour contrôler les attributs
    ds.name = "dummy/repo"
    ds.split = "train"
    ds.field = "missing"  # volontairement absent
    ds.limit = 2
    ds.data = None

    with pytest.raises(KeyError) as exc:
        ds.download()
    # message d'erreur informatif
    assert "introuvable" in str(exc.value)
    # data n'a pas été remplie
    assert ds.data is None


def test_download_zero_limit(monkeypatch):
    """
    limit=0 : la boucle ne s'exécute pas, DataFrame vide est créé.
    """

    def fake_load_dataset(name, split=None, streaming=False, **kwargs):
        # même s'il y a des données, limit=0 empêchera toute itération
        rows = [{"text": "a"}, {"text": "b"}]
        return make_stream_iter(rows)

    monkeypatch.setattr("dataset.load_dataset", fake_load_dataset)

    ds = Dataset.__new__(Dataset)
    ds.name = "dummy/repo"
    ds.split = "train"
    ds.field = "text"
    ds.limit = 0
    ds.data = None

    # exécute
    ds.download()

    # DataFrame vide attendu
    assert isinstance(ds.data, pd.DataFrame)
    assert len(ds.data) == 0
    # Quand on construit un DataFrame à partir d'une liste vide, il n'a pas de colonnes
    assert list(ds.data.columns) == []
