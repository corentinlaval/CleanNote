from cleanote.data_downloader import DataDownloader
from cleanote.types import Context


def test_data_downloader_stub_fetch_returns_empty_iterable():
    d = DataDownloader(
        hf_dataset_name="dummy/dataset",
        split="train",
        text_field="text",
        limit=5,
    )
    out = list(d.fetch(Context(run_id="t")))
    assert out == []  # itérateur vide


def test_data_downloader_stub_attributes_are_set():
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
