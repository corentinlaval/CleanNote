# cleanote/data_downloader.py
from typing import Iterable, Optional
from .types import Doc, Context


def _get_by_path(row: dict, path: str):
    """Allows dot-notation to extract nested fields, e.g. 'record.text'."""
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
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.text_field = text_field
        self.limit = limit

    def fetch(self, ctx: Context) -> Iterable[Doc]:
        """Download a Hugging Face dataset and yield Doc objects."""
        from datasets import load_dataset

        print(
            f"[DataDownloader] Loading dataset '{self.hf_dataset_name}' "
            f"(split='{self.split}', text_field='{self.text_field}')..."
        )

        ds = load_dataset(path=self.hf_dataset_name, split=self.split)

        count = 0
        for i, row in enumerate(ds):
            text = None
            try:
                text = _get_by_path(dict(row), self.text_field)
            except Exception:
                pass

            if not isinstance(text, str):
                continue

            if count == 0:
                print(f"[DataDownloader] First row example: {row}")

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

        print(f"[DataDownloader] Finished. Yielded {count} documents.")
