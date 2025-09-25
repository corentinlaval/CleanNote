import pandas as pd
import pytest

from cleanote.dataset import Dataset


def make_stream_iter(rows):
    def _iter():
        for r in rows:
            yield r

    return _iter()


def test_download_happy_path(monkeypatch):
    captured = {}

    def fake_load_dataset(name, split=None, streaming=False, **kwargs):
        captured["name"] = name
        captured["split"] = split
        captured["streaming"] = streaming
        rows = [{"text": f"note {i}", "other": i} for i in range(5)]
        return make_stream_iter(rows)

    monkeypatch.setattr("cleanote.dataset.load_dataset", fake_load_dataset)

    ds = Dataset(name="dummy/repo", split="train", field="text", limit=3)

    assert captured["name"] == "dummy/repo"
    assert captured["split"] == "train"
    assert captured["streaming"] is True

    assert isinstance(ds.data, pd.DataFrame)
    assert len(ds.data) == 3
    assert list(ds.data.columns) == ["index", "text"]
    assert ds.data.loc[0, "index"] == 0
    assert ds.data.loc[0, "text"] == "note 0"
    assert ds.data.loc[2, "text"] == "note 2"


def test_download_missing_field_raises(monkeypatch):
    def fake_load_dataset(name, split=None, streaming=False, **kwargs):
        rows = [{"text": "ok"}, {"text": "ok 2"}]
        return make_stream_iter(rows)

    monkeypatch.setattr("cleanote.dataset.load_dataset", fake_load_dataset)

    ds = Dataset.__new__(Dataset)
    ds.name = "dummy/repo"
    ds.split = "train"
    ds.field = "missing"
    ds.limit = 2
    ds.data = None

    with pytest.raises(KeyError) as exc:
        ds.download()
    assert "introuvable" in str(exc.value)
    assert ds.data is None


def test_download_zero_limit(monkeypatch):
    def fake_load_dataset(name, split=None, streaming=False, **kwargs):
        rows = [{"text": "a"}, {"text": "b"}]
        return make_stream_iter(rows)

    monkeypatch.setattr("cleanote.dataset.load_dataset", fake_load_dataset)

    ds = Dataset.__new__(Dataset)
    ds.name = "dummy/repo"
    ds.split = "train"
    ds.field = "text"
    ds.limit = 0
    ds.data = None

    ds.download()

    assert isinstance(ds.data, pd.DataFrame)
    assert len(ds.data) == 0
    # DataFrame vide ⇒ pas de colonnes
    assert list(ds.data.columns) == []
