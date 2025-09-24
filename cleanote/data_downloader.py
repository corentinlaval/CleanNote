# cleanote/data_downloader.py
from typing import Iterable, Optional
from .types import Doc, Context


def _get_by_path(row: dict, path: str):
    """Permet d'utiliser une dot-notation: ex. 'record.text'."""
    cur = row
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


class DataDownloader:
    def __init__(
        self,
        hf_dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        limit: Optional[int] = None,
        revision: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.text_field = text_field
        self.limit = limit
        self.revision = revision
        self.streaming = streaming

    def fetch(self, ctx: Context) -> Iterable[Doc]:
        """Télécharge un dataset Hugging Face et renvoie un flux de Doc."""
        from datasets import (
            load_dataset,
        )  # import local pour ne pas forcer la dépendance

        ds = load_dataset(
            path=self.hf_dataset_name,
            split=self.split,
            revision=self.revision,
            streaming=self.streaming,
        )

        count = 0
        iterator = iter(ds) if self.streaming else (row for row in ds)

        for i, row in enumerate(iterator):
            text = None
            try:
                text = _get_by_path(dict(row), self.text_field)
            except Exception:
                pass

            if not isinstance(text, str):
                continue

            yield Doc(
                id=f"{self.hf_dataset_name}:{self.split}:{i}",
                text=text,
                meta={
                    "source": "hf",
                    "dataset": self.hf_dataset_name,
                    "split": self.split,
                },
            )

            count += 1
            if self.limit is not None and count >= self.limit:
                break
