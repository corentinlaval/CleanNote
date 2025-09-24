# tests/test_downloader.py
import sys
import types
from cleanote.data_downloader import DataDownloader
from cleanote.types import Context


def test_data_downloader_attributes_are_set():
    d = DataDownloader(
        hf_dataset_name="org/name",
        split="valid",
        text_field="note",
        limit=1,
    )
    assert d.hf_dataset_name == "org/name"
    assert d.split == "valid"
    assert d.text_field == "note"
    assert d.limit == 1


def test_data_downloader_fetch_empty_when_text_field_missing(monkeypatch):
    # On fabrique un faux module 'datasets' avec un load_dataset qui renvoie des lignes
    def fake_load_dataset(path, split, revision=None, streaming=False):
        # colonnes ne contenant PAS 'note' -> fetch doit sauter ces lignes => iterable vide
        return [{"wrong": "x"}, {"foo": "bar"}]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    d = DataDownloader(
        hf_dataset_name="dummy/ds",
        split="train",
        text_field="note",  # inexistante dans les rows ci-dessus
        limit=5,
    )
    out = list(d.fetch(Context(run_id="t")))
    assert out == []  # rien à retourner car le champ demandé manque


def test_data_downloader_fetch_returns_docs_and_respects_limit(monkeypatch):
    # Faux dataset avec la bonne colonne 'text'
    rows = [
        {"text": "hello   world"},
        {"text": "second row"},
        {"text": "third row"},
    ]

    def fake_load_dataset(path, split):
        # Simule un iterable (streaming=False => on itère quand même dessus dans notre code)
        return rows

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    d = DataDownloader(
        hf_dataset_name="ag_news",  # peu importe, on mocke
        split="train",
        text_field="text",
        limit=2,  # on ne prend que 2 lignes
    )
    docs = list(d.fetch(Context(run_id="t")))
    assert len(docs) == 2
    assert docs[0].text == "hello   world"
    assert docs[0].id.startswith("ag_news:train:")
    assert docs[0].meta["source"] == "hf"
