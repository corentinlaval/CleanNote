from typing import Iterable
from .types import Doc, Context


class DataDownloader:
    def __init__(
        self,
        hf_dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        limit: int | None = None,
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.text_field = text_field
        self.limit = limit

    def fetch(self, ctx: Context) -> Iterable[Doc]:
        """Stub: retourne bientôt un flux de Doc. Pour l’instant, yield rien."""
        if False:
            yield Doc(id="placeholder", text="")  # pour typer
        return
