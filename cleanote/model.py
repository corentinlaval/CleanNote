from __future__ import annotations

from typing import Any, Dict, List, Tuple
import copy
import inspect
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    pipeline,
)


class Model:
    """
    Model HF générique (DataFrame-only) avec routage des kwargs :
      - kwargs "génération" (ex: max_new_tokens, temperature, top_p, …) → passés au pipeline à l'appel
      - kwargs "model_*"     (ex: model_quantization_config=..., model_device_map="auto", model_torch_dtype=...)
      - kwargs "tokenizer_*" (ex: tokenizer_use_fast=True, tokenizer_padding_side="left")
      - kwargs "pipeline_*"  (ex: pipeline_device_map="auto", pipeline_return_full_text=False, batch_size=...)

    Exemple d'usage plus bas avec Llama3-OpenBioLLM 8B en 8-bit.
    """

    def __init__(self, name: str, task: str = "text-generation", **kwargs: Any):
        self.name = name
        self.task = task

        (
            self.pipeline_kwargs,
            self.generation_kwargs,
            self.model_kwargs,
            self.tokenizer_kwargs,
        ) = self._split_kwargs(kwargs)

        # Politesse: si tu donnes max_new_tokens + max_length, on supprime max_length
        if (
            "max_new_tokens" in self.generation_kwargs
            and "max_length" in self.generation_kwargs
        ):
            self.generation_kwargs.pop("max_length", None)

        self._tokenizer = None
        self._model = None
        self._pipe = None
        self.load()

    # -------- utils: détection des clés --------
    @staticmethod
    def _gen_keys() -> set:
        return set(GenerationConfig().to_dict().keys())

    @staticmethod
    def _pipe_keys(task: str) -> set:
        sig = inspect.signature(pipeline)
        common = {
            "device",
            "device_map",
            "framework",
            "batch_size",
            "torch_dtype",
            "return_full_text",
        }
        return {p.name for p in sig.parameters.values()} | common

    def _split_kwargs(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        gen_keys = self._gen_keys()
        pipe_keys = self._pipe_keys(self.task)

        pipeline_kwargs: Dict[str, Any] = {}
        generation_kwargs: Dict[str, Any] = {}
        model_kwargs: Dict[str, Any] = {}
        tokenizer_kwargs: Dict[str, Any] = {}

        for k, v in kwargs.items():
            if k.startswith("model_"):
                model_kwargs[k[len("model_") :]] = v
            elif k.startswith("tokenizer_"):
                tokenizer_kwargs[k[len("tokenizer_") :]] = v
            elif k.startswith("pipeline_"):
                pipeline_kwargs[k[len("pipeline_") :]] = v
            elif k in gen_keys:
                generation_kwargs[k] = v
            elif k in pipe_keys:
                pipeline_kwargs[k] = v
            else:
                # Par défaut: on considère que c'est un paramètre de génération
                generation_kwargs[k] = v

        return pipeline_kwargs, generation_kwargs, model_kwargs, tokenizer_kwargs

    # -------- chargement --------
    def load(self) -> None:
        if self._pipe is not None:
            return

        print(f"[Model] Loading model '{self.name}' for task '{self.task}'...")

        # Tokenizer
        if "use_fast" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["use_fast"] = True
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.name, **self.tokenizer_kwargs
        )

        # pad_token pour certains modèles (OPT…) si absent
        if (
            self._tokenizer.pad_token_id is None
            and self._tokenizer.eos_token_id is not None
        ):
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Model head selon la task
        model_cls = (
            AutoModelForSeq2SeqLM
            if self.task in {"text2text-generation", "summarization", "translation"}
            else AutoModelForCausalLM
        )

        # Valeurs par défaut "safe"
        self.model_kwargs.setdefault("low_cpu_mem_usage", True)
        self.model_kwargs.setdefault("use_safetensors", True)

        # Chargement modèle (supporte quantization_config, device_map, torch_dtype, etc.)
        self._model = model_cls.from_pretrained(self.name, **self.model_kwargs)

        # Pipeline
        if (
            "device" not in self.pipeline_kwargs
            and "device_map" not in self.pipeline_kwargs
        ):
            self.pipeline_kwargs["device"] = -1  # CPU par défaut
        self._pipe = pipeline(
            self.task,
            model=self._model,
            tokenizer=self._tokenizer,
            **self.pipeline_kwargs,
        )

        print("[Model] Load completed.")

    # -------- inférence --------
    def _apply_to_texts(self, texts: List[str], prompt: str) -> List[str]:
        if self._pipe is None:
            self.load()

        infer_kwargs = dict(self.generation_kwargs)
        # Pour 'text-generation', renvoyer seulement la génération (pas l'input)
        if (
            self.task == "text-generation"
            and "return_full_text" not in infer_kwargs
            and "return_full_text" not in self.pipeline_kwargs
        ):
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

    def run(self, dataset, prompt: str, output_col: str | None = None):
        """DataFrame-only: ajoute une colonne avec la réponse du modèle."""
        if not hasattr(dataset, "data"):
            raise ValueError("Le dataset fourni n'a pas d'attribut 'data'.")
        if not isinstance(dataset.data, pd.DataFrame):
            raise TypeError("Le dataset.data doit être un pandas.DataFrame.")
        if not hasattr(dataset, "field"):
            raise ValueError("Le dataset DataFrame doit définir l'attribut 'field'.")
        if dataset.field not in dataset.data.columns:
            raise KeyError(
                f"Colonne '{dataset.field}' introuvable. Colonnes: {list(dataset.data.columns)}"
            )

        df = dataset.data.copy()
        outs = self._apply_to_texts(df[dataset.field].astype(str).tolist(), prompt)

        safe_name = self.name.replace("/", "_").replace("-", "_").replace(".", "_")
        if output_col is None:
            output_col = f"{dataset.field}__{safe_name}"

        base, i = output_col, 1
        while output_col in df.columns:
            output_col = f"{base}_{i}"
            i += 1

        df[output_col] = outs

        result_ds = copy.copy(dataset)
        result_ds.data = df
        result_ds.last_output_col = output_col
        if hasattr(result_ds, "limit"):
            result_ds.limit = len(df)
        if hasattr(result_ds, "name"):
            result_ds.name = f"{getattr(dataset, 'name', 'dataset')}__{safe_name}"
        return result_ds
