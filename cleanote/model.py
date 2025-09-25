from __future__ import annotations

from typing import Any, Dict, List
import copy

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Model:
    """
    Modèle Hugging Face générique :
      - name  : repo_id (ex: 'gpt2', 'google/flan-t5-small', ...)
      - task  : tâche du pipeline (ex: 'text-generation', 'text2text-generation', ...)
      - params: kwargs passés au pipeline / génération (ex: max_new_tokens, temperature, ...)
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
        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        # Pour les tâches autres que causal LM, tu peux adapter ici le type de modèle
        self._model = AutoModelForCausalLM.from_pretrained(self.name)
        self._pipe = pipeline(self.task, model=self._model, tokenizer=self._tokenizer)
        print("[Model] Load completed.")

    def _apply_to_texts(self, texts: List[str], prompt: str) -> List[str]:
        """
        Applique le modèle à une liste de textes et retourne UNIQUEMENT la sortie (réponse au prompt).
        - Pour 'text-generation', on concatène prompt + texte et on isole la partie générée.
        - Pour autres tâches, on normalise pour obtenir une chaîne finale.
        """
        if self._pipe is None:
            self.load()

        outputs: List[str] = []
        infer_kwargs = dict(self.params)

        for txt in texts:
            inp = f"{prompt}\n\n{txt}".strip()

            result = self._pipe(inp, **infer_kwargs)

            if self.task == "text-generation":
                # Liste de dicts avec 'generated_text'
                gen = result[0].get("generated_text", "")
                # Heuristique: retirer l'input de la sortie si présent
                if gen.startswith(inp):
                    gen = gen[len(inp) :].lstrip("\n")
                outputs.append(gen)
            else:
                # Normalisation générique
                if isinstance(result, list):
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
        Applique le modèle au dataset et RENVOIE un NOUVEAU dataset du même type
        (dict ou DataFrame) où les valeurs sont REMPLACÉES par la réponse au prompt.
        """
        if not hasattr(dataset, "data"):
            raise ValueError("Le dataset fourni n'a pas d'attribut 'data'.")

        # Copier l'objet sans relancer d'__init__
        result_ds = copy.copy(dataset)

        # --- Cas dict {index: texte} ---
        if isinstance(dataset.data, dict):
            items = sorted(dataset.data.items(), key=lambda kv: kv[0])
            indices = [i for i, _ in items]
            texts = [str(t) for _, t in items]

            outs = self._apply_to_texts(texts, prompt)
            result_ds.data = {i: o for i, o in zip(indices, outs)}

            if hasattr(result_ds, "limit"):
                result_ds.limit = len(result_ds.data)
            if hasattr(result_ds, "name"):
                result_ds.name = f"{getattr(dataset, 'name', 'dataset')}__{self.name}"
            return result_ds

        # --- Cas DataFrame ---
        try:
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

            # Remplacement pur par le résultat du prompt
            df[dataset.field] = outs
            result_ds.data = df

            if hasattr(result_ds, "limit"):
                result_ds.limit = len(df)
            if hasattr(result_ds, "name"):
                result_ds.name = f"{getattr(dataset, 'name', 'dataset')}__{self.name}"
            return result_ds

        # Type non géré
        raise TypeError(
            "Type de dataset.data non pris en charge. Utilise un dict {index: texte} "
            "ou un pandas.DataFrame avec la colonne 'field'."
        )
