from datasets import load_dataset
import pandas as pd
from itertools import islice


class Dataset:
    def __init__(self, name: str, split: str, field: str, limit: int):
        self.name = name
        self.split = split
        self.field = field
        self.limit = limit
        self.data = None  # DataFrame (index, texte)

        self.download()

    def download(self):
        print(
            f"[Dataset] Streaming {self.limit} rows from '{self.name}' ({self.split})..."
        )

        # streaming=True => lit ligne par ligne à distance
        ds_iter = load_dataset(self.name, split=self.split, streaming=True)

        rows = []
        for i, row in islice(enumerate(ds_iter), self.limit):
            if self.field not in row:
                raise KeyError(
                    f"Le champ '{self.field}' est introuvable. "
                    f"Champs disponibles: {list(row.keys())}"
                )
            rows.append({"index": i, self.field: row[self.field]})

        self.data = pd.DataFrame(rows)

        print(f"[Dataset] Download completed. Loaded {len(self.data)} rows.")
