from datasets import load_dataset
import pandas as pd


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
            f"[Dataset] Downloading {self.limit} rows from '{self.name}' ({self.split})..."
        )

        dataset = load_dataset(self.name, split=f"{self.split}[:{self.limit}]")
        texts = dataset[self.field]

        # DataFrame avec deux colonnes : index + texte
        self.data = pd.DataFrame({"index": range(len(texts)), self.field: texts})

        print(f"[Dataset] Download completed. Loaded {len(self.data)} rows.")
