from __future__ import annotations

from typing import Any, Dict, List
import copy

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Model:
    """
    Model générique basé sur Hugging Face:
      - name : repo_id du modèle (ex: 'gpt2', 'meta-llama/Llama-3-8B-Instruct', etc.)
      - task : tâche du pipeline (ex: 'text-generation', 'fill-mask', 'text2text-generation'…)
      - params : kwargs passés au pipeline et/ou à la génération (max_new_tokens, temperature, etc.)
    """

    def __init__(self, name: str, task: str = "text-generation", **params: Any):
        self.name = name
        self.task = task
        self.params: Dict[str, Any] = params or {}

        self._tokenizer = None
        self._model = None
        self._pipe = None

        self.load()

    def load(self) -> None:
        """Télécharge le modèle & le tokenizer depuis Hugging Face et prépare le pipeline."""
        if self._pipe is not None:
            return  # déjà chargé

        print(f"[Model] Loading model '{self.name}' for task '{self.task}'...")
        # Vous pouvez ajuster device_map="auto" si vous avez accelerate+GPU
        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        self._model = AutoModelForCausalLM.from_pretrained(self.name)
        self._pipe = pipeline(self.task, model=self._model, tokenizer=self._tokenizer)

        print("[Model] Load completed.")

    def _apply_to_texts(self, texts: List[str], prompt: str) -> List[str]:
        """
        Applique le modèle à une liste de textes.
        Pour 'text-generation', on concatène prompt + texte et on récupère la génération.
        """
        if self._pipe is None:
            self.load()

        outputs: List[str] = []
        # Paramètres de génération/pipeline (ex: max_new_tokens=64)
        infer_kwargs = dict(self.params)

        # Pour 'text-generation', on va concaténer prompt et texte.
        for txt in texts:
            if self.task == "text-generation":
                inp = f"{prompt}\n\n{txt}"
                result = self._pipe(inp, **infer_kwargs)
                # result est une liste de dicts avec 'generated_text'
                gen = result[0].get("generated_text", "")
                # Si le pipeline renvoie tout le texte, on isole la partie génération
                # en supprimant l'entrée initiale (simple heuristique)
                if gen.startswith(inp):
                    gen = gen[len(inp) :].lstrip("\n")
                outputs.append(gen)
            else:
                # Cas générique : on passe le prompt + texte dans un seul input string
                inp = f"{prompt}\n\n{txt}".strip()
                result = self._pipe(inp, **infer_kwargs)
                # Normaliser différents formats de sorties
                if isinstance(result, list):
                    # Prendre le champ text s'il existe, sinon str(result[0])
                    val = (
                        result[0].get("generated_text")
                        or result[0].get("summary_text")
                        or result[0].get("answer")
                        or result[0].get("sequence")
                        or str(result[0])
                    )
                else:
                    val = str(result)
                outputs.append(val)

        return outputs

    def run(self, dataset, prompt: str):
        """
        Applique le modèle au dataset et renvoie un NOUVEAU dataset du même type
        (dict ou DataFrame), sans re-télécharger depuis Hugging Face.
        - Si dataset.data est un dict {index: texte}, on renvoie un dict {index: sortie}.
        - Si c'est un DataFrame avec colonnes ['index', dataset.field], on remplace la colonne `field` par la sortie.
        """
        if not hasattr(dataset, "data"):
            raise ValueError("Le dataset fourni n'a pas d'attribut 'data'.")

        # Copier l'objet sans appeler __init__ (évite un re-download)
        result_ds = copy.copy(dataset)

        # dict {index: texte}
        if isinstance(dataset.data, dict):
            # Assurer un ordre stable par index
            items = sorted(dataset.data.items(), key=lambda kv: kv[0])
            indices = [i for i, _ in items]
            texts = [str(t) for _, t in items]

            outs = self._apply_to_texts(texts, prompt)
            result_ds.data = {i: o for i, o in zip(indices, outs)}
            # mettre à jour limit si présent
            if hasattr(result_ds, "limit"):
                result_ds.limit = len(result_ds.data)
            # optionnel: annoter le nom
            if hasattr(result_ds, "name"):
                result_ds.name = f"{getattr(dataset, 'name', 'dataset')}__{self.name}"

            return result_ds

        # DataFrame (pandas)
        try:
            import pandas as pd  # noqa

            is_df = hasattr(dataset.data, "iloc") and hasattr(dataset.data, "columns")
        except Exception:
            is_df = False

        if is_df:
            if not hasattr(dataset, "field"):
                raise ValueError(
                    "Le dataset de type DataFrame doit définir l'attribut 'field'."
                )

            df = dataset.data.copy()
            if dataset.field not in df.columns:
                raise KeyError(
                    f"Colonne '{dataset.field}' introuvable dans le DataFrame. "
                    f"Colonnes dispo: {list(df.columns)}"
                )

            texts = df[dataset.field].astype(str).tolist()
            outs = self._apply_to_texts(texts, prompt)

            # Remplace la colonne par la sortie du modèle
            df[dataset.field] = outs
            result_ds.data = df
            if hasattr(result_ds, "limit"):
                result_ds.limit = len(df)
            if hasattr(result_ds, "name"):
                result_ds.name = f"{getattr(dataset, 'name', 'dataset')}__{self.name}"

            return result_ds

        # Sinon, type non géré
        raise TypeError(
            "Type de dataset.data non pris en charge. Utilise un dict {index: texte} "
            "ou un pandas.DataFrame avec la colonne 'field'."
        )
