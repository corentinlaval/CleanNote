# tests/test_data_downloader.py

from cleanote.data_downloader import FSDownloader
from cleanote.types import Context


def test_fs_downloader_reads_files(tmp_path):
    # Création de deux fichiers temporaires
    file1 = tmp_path / "a.txt"
    file1.write_text("hello")
    file2 = tmp_path / "b.txt"
    file2.write_text("world")

    # Instanciation du downloader
    d = FSDownloader(folder=tmp_path)

    # Récupération des documents
    docs = list(d.fetch(Context(run_id="t")))

    # Vérifications
    assert len(docs) == 2
    assert [doc.text for doc in docs] == ["hello", "world"]
    assert docs[0].id == "a"
    assert docs[1].id == "b"
    assert all(doc.meta["source"] == "fs" for doc in docs)
