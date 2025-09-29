from __future__ import annotations
from typing import Any, Dict, Tuple
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import copy
import pandas as pd


def _ensure_pad_token(tok):
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token


_PIPELINE_KEYS = {
    "device",
    "device_map",
    "framework",
    "batch_size",
    "return_full_text",
    "torch_dtype",
    "dtype",
}


def _normalize_dtypes(d: Dict[str, Any]) -> None:
    if "dtype" in d and "torch_dtype" not in d:
        d["torch_dtype"] = d.pop("dtype")


def _split_kwargs_simple(
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    gen_keys = set(GenerationConfig().to_dict().keys())

    pipeline_kwargs: Dict[str, Any] = {}
    generation_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    tokenizer_kwargs: Dict[str, Any] = {}

    for k, v in kwargs.items():
        if k.startswith("model_"):
            model_kwargs[k[len("model_") :]] = v
        elif k.startswith("tokenizer_"):
            tokenizer_kwargs[k[len("tokenizer_") :]] = v
        elif k in gen_keys:
            generation_kwargs[k] = v
        elif k in _PIPELINE_KEYS:
            pipeline_kwargs[k] = v
        else:
            generation_kwargs[k] = v

    _normalize_dtypes(pipeline_kwargs)
    _normalize_dtypes(model_kwargs)

    return pipeline_kwargs, generation_kwargs, model_kwargs, tokenizer_kwargs


class Model:
    """Wrapper simple pour modèles de génération de texte (text-generation uniquement)."""

    def __init__(self, name: str, task: str = "text-generation", **kwargs: Any):
        if task != "text-generation":
            raise ValueError("Only 'text-generation' is supported in this Model.")

        self.name = name
        self.task = task

        (
            self.pipeline_kwargs,
            self.generation_kwargs,
            self.model_kwargs,
            self.tokenizer_kwargs,
        ) = _split_kwargs_simple(kwargs)

        if (
            "max_new_tokens" in self.generation_kwargs
            and "max_length" in self.generation_kwargs
        ):
            self.generation_kwargs.pop("max_length", None)

        self.generation_kwargs.setdefault("do_sample", False)
        self.generation_kwargs.setdefault("temperature", 0.0)
        self.generation_kwargs.setdefault("top_p", 1.0)
        self.pipeline_kwargs.setdefault("return_full_text", False)

        if (
            "device" not in self.pipeline_kwargs
            and "device_map" not in self.pipeline_kwargs
        ):
            self.pipeline_kwargs["device"] = -1

        self._tokenizer = None
        self._model = None
        self._pipe = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        print(f"[Model] Loading model '{self.name}' for task '{self.task}'...")

        if "use_fast" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["use_fast"] = True
        tok = AutoTokenizer.from_pretrained(self.name, **self.tokenizer_kwargs)
        _ensure_pad_token(tok)

        self.model_kwargs.setdefault("low_cpu_mem_usage", True)
        self.model_kwargs.setdefault("use_safetensors", True)
        mdl = AutoModelForCausalLM.from_pretrained(self.name, **self.model_kwargs)

        self._pipe = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            **self.pipeline_kwargs,
        )
        print("[Model] Load completed.")

    def run(self, dataset, prompt: str, output_col: str | None = None, **gen_overrides):
        """Applique le modèle sur dataset[field] et ajoute une colonne avec la sortie."""
        if self._pipe is None:
            self.load()

        if not hasattr(dataset, "data"):
            raise ValueError("Le dataset fourni n'a pas d'attribut 'data'.")
        if not isinstance(dataset.data, pd.DataFrame):
            raise TypeError("Le dataset.data doit être un pandas.DataFrame.")
        if not hasattr(dataset, "field"):
            raise ValueError("Le dataset doit définir l'attribut 'field'.")
        if dataset.field not in dataset.data.columns:
            raise KeyError(f"Colonne '{dataset.field}' introuvable.")

        df = dataset.data.copy()
        texts = df[dataset.field].astype(str).tolist()

        # fusion defaults init + overrides appel
        infer_kwargs = {**self.generation_kwargs, **gen_overrides}

        outs = []
        for txt in texts:
            inp = f"{prompt}\n\n{txt}".strip()
            result = self._pipe(inp, **infer_kwargs)
            # robustifier l'extraction du texte généré
            if isinstance(result, list) and result:
                outs.append(result[0].get("generated_text", ""))
            elif isinstance(result, dict):
                outs.append(result.get("generated_text", ""))
            else:
                outs.append(str(result))

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
        return result_ds
