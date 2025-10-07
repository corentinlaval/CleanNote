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

        # Chargement en streaming
        ds_iter = load_dataset(self.name, split=self.split, streaming=True)

        rows = []
        detected_id_field = None

        # On lit la première ligne pour détecter les clés disponibles
        first_row = next(iter(ds_iter))
        ds_iter = load_dataset(
            self.name, split=self.split, streaming=True
        )  # on recharge pour repartir du début

        # Détection d'une colonne "id" (nom contenant 'id')
        for key in first_row.keys():
            if "id" in key.lower():
                detected_id_field = key
                break

        if detected_id_field:
            print(
                f"[Dataset] Colonne ID détectée automatiquement : '{detected_id_field}'"
            )
        else:
            print("[Dataset] Aucune colonne ID détectée, un index sera créé.")

        for i, row in islice(enumerate(ds_iter), self.limit):
            if self.field not in row:
                raise KeyError(
                    f"Le champ '{self.field}' est introuvable. "
                    f"Champs disponibles: {list(row.keys())}"
                )

            # On utilise l'ID détecté si présent, sinon on crée un index
            row_id = row.get(detected_id_field, i) if detected_id_field else i
            rows.append({"index": row_id, self.field: row[self.field]})

        self.data = pd.DataFrame(rows)

        print(f"[Dataset] Download completed. Loaded {len(self.data)} rows.")
