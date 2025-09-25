from __future__ import annotations

from typing import Any, Dict, List
import copy

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)


class Model:
    """
    Modèle Hugging Face (DataFrame-only) :
      - name  : repo_id (ex: 'gpt2', 'google/flan-t5-small', ...)
      - task  : 'text-generation' (causal LM) ou 'text2text-generation' (seq2seq), etc.
      - params: kwargs passés au pipeline/génération (ex: max_new_tokens, temperature, ...)
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

        # Choix du type de modèle selon la tâche
        if self.task in {"text2text-generation", "summarization", "translation"}:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.name)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.name)

        self._pipe = pipeline(self.task, model=self._model, tokenizer=self._tokenizer)
        print("[Model] Load completed.")

    def _apply_to_texts(self, texts: List[str], prompt: str) -> List[str]:
        """
        Applique le modèle à une liste de textes et retourne UNIQUEMENT la réponse au prompt.
        - 'text-generation' : concatène prompt + texte, isole la partie générée.
        - 'text2text-generation' et assimilés : retourne la séquence générée.
        """
        if self._pipe is None:
            self.load()

        outputs: List[str] = []
        infer_kwargs = dict(self.params)

        for txt in texts:
            inp = f"{prompt}\n\n{txt}".strip()
            result = self._pipe(inp, **infer_kwargs)

            if self.task == "text-generation":
                gen = result[0].get("generated_text", "")
                if gen.startswith(inp):
                    gen = gen[len(inp) :].lstrip("\n")
                outputs.append(gen)
            else:
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
        Applique le modèle au dataset (DataFrame uniquement) et renvoie un NOUVEAU dataset
        où la colonne `dataset.field` est REMPLACÉE par la réponse au prompt.
        """
        if not hasattr(dataset, "data"):
            raise ValueError("Le dataset fourni n'a pas d'attribut 'data'.")

        # Vérifier que c'est bien un DataFrame
        is_df = hasattr(dataset.data, "iloc") and hasattr(dataset.data, "columns")
        if not is_df:
            raise TypeError("Le dataset.data doit être un pandas.DataFrame.")

        if not hasattr(dataset, "field"):
            raise ValueError("Le dataset DataFrame doit définir l'attribut 'field'.")

        df = dataset.data.copy()
        if dataset.field not in df.columns:
            raise KeyError(
                f"Colonne '{dataset.field}' introuvable dans le DataFrame. "
                f"Colonnes dispo: {list(df.columns)}"
            )

        texts = df[dataset.field].astype(str).tolist()
        outs = self._apply_to_texts(texts, prompt)

        # Remplacement pur (la colonne 'index' reste intacte)
        df[dataset.field] = outs

        # Retourner une copie superficielle du Dataset avec le DF mis à jour
        result_ds = copy.copy(dataset)
        result_ds.data = df
        if hasattr(result_ds, "limit"):
            result_ds.limit = len(df)
        if hasattr(result_ds, "name"):
            result_ds.name = f"{getattr(dataset, 'name', 'dataset')}__{self.name}"

        return result_ds
