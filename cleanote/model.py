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
    def __init__(self, name: str, task: str = "text-generation", **params: Any):
        self.name = name
        self.task = task
        self.params: Dict[str, Any] = params or {}

        # évite l’avertissement HF max_length vs max_new_tokens
        if "max_new_tokens" in self.params and "max_length" in self.params:
            self.params.pop("max_length", None)

        self._tokenizer = None
        self._model = None
        self._pipe = None
        self.load()

    def load(self) -> None:
        if self._pipe is not None:
            return
        print(f"[Model] Loading model '{self.name}' for task '{self.task}'...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=True)
        if (
            self._tokenizer.pad_token_id is None
            and self._tokenizer.eos_token_id is not None
        ):
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if self.task in {"text2text-generation", "summarization", "translation"}:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.name, low_cpu_mem_usage=True, use_safetensors=True
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.name, low_cpu_mem_usage=True, use_safetensors=True
            )

        # device=-1 => CPU ; si GPU dispo: device=0
        self._pipe = pipeline(
            self.task, model=self._model, tokenizer=self._tokenizer, device=-1
        )
        print("[Model] Load completed.")

    def _apply_to_texts(self, texts: List[str], prompt: str) -> List[str]:
        if self._pipe is None:
            self.load()

        infer_kwargs = dict(self.params)
        # pour text-generation, ne pas renvoyer le texte complet d’entrée
        if self.task == "text-generation" and "return_full_text" not in infer_kwargs:
            infer_kwargs["return_full_text"] = False

        outs: List[str] = []
        for txt in texts:
            inp = f"{prompt}\n\n{txt}".strip()
            result = self._pipe(inp, **infer_kwargs)

            if self.task == "text-generation":
                outs.append(result[0].get("generated_text", ""))
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
                outs.append(val)
        return outs

    def _safe_col_name(self, base: str) -> str:
        return base.replace("/", "_").replace("-", "_").replace(".", "_")

    def run(self, dataset, prompt: str, output_col: str | None = None):
        """
        Applique le modèle et **ajoute** une nouvelle colonne contenant la réponse,
        en conservant la colonne source `dataset.field`.
        """
        if not hasattr(dataset, "data"):
            raise ValueError("Le dataset fourni n'a pas d'attribut 'data'.")
        if not (hasattr(dataset.data, "iloc") and hasattr(dataset.data, "columns")):
            raise TypeError("Le dataset.data doit être un pandas.DataFrame.")
        if not hasattr(dataset, "field"):
            raise ValueError("Le dataset DataFrame doit définir l'attribut 'field'.")

        df = dataset.data.copy()
        if dataset.field not in df.columns:
            raise KeyError(
                f"Colonne '{dataset.field}' introuvable. Colonnes: {list(df.columns)}"
            )

        texts = df[dataset.field].astype(str).tolist()
        outs = self._apply_to_texts(texts, prompt)

        # nom de colonne de sortie
        if output_col is None:
            output_col = f"{dataset.field}__{self._safe_col_name(self.name)}"
        # éviter d’écraser si la colonne existe déjà
        base = output_col
        i = 1
        while output_col in df.columns:
            output_col = f"{base}_{i}"
            i += 1

        df[output_col] = outs

        result_ds = copy.copy(dataset)
        result_ds.data = df
        if hasattr(result_ds, "limit"):
            result_ds.limit = len(df)
        if hasattr(result_ds, "name"):
            result_ds.name = f"{getattr(dataset, 'name', 'dataset')}__{self._safe_col_name(self.name)}"
        # garder trace de la dernière colonne sortie (pratique pour l’inspection)
        result_ds.last_output_col = output_col  # attribut pratique
        return result_ds
