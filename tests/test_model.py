from __future__ import annotations
from typing import Any, Dict, Tuple
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import copy
import pandas as pd
import json
import re


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
    # Si "dtype" est present et "torch_dtype" absent, normaliser vers "torch_dtype"
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
            # Par défaut, tout va dans les kwargs de génération
            generation_kwargs[k] = v

    _normalize_dtypes(pipeline_kwargs)
    _normalize_dtypes(model_kwargs)

    return pipeline_kwargs, generation_kwargs, model_kwargs, tokenizer_kwargs


def _extract_json_block(text: str) -> dict:
    """
    Extrait le premier bloc JSON trouvé dans `text` et le parse en dict.
    Retourne {} si rien de valide n'est trouvé.
    """
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    return {}


class Model:
    """Wrapper simple pour modèles de génération de texte (text-generation uniquement)."""

    def __init__(
        self, name: str, task: str = "text-generation", prompt: str = "", **kwargs: Any
    ):
        if task != "text-generation":
            raise ValueError("Only 'text-generation' is supported in this Model.")

        self.name = name
        self.task = task
        self.prompt = prompt

        (
            self.pipeline_kwargs,
            self.generation_kwargs,
            self.model_kwargs,
            self.tokenizer_kwargs,
        ) = _split_kwargs_simple(kwargs)

        # Si max_new_tokens est fourni, ignorer max_length pour éviter les conflits
        if (
            "max_new_tokens" in self.generation_kwargs
            and "max_length" in self.generation_kwargs
        ):
            self.generation_kwargs.pop("max_length", None)

        # Valeurs par défaut raisonnables
        self.generation_kwargs.setdefault("do_sample", False)
        self.generation_kwargs.setdefault("temperature", 0.0)
        self.generation_kwargs.setdefault("top_p", 1.0)
        self.pipeline_kwargs.setdefault("return_full_text", False)

        # device de pipeline par défaut = CPU
        if (
            "device" not in self.pipeline_kwargs
            and "device_map" not in self.pipeline_kwargs
        ):
            self.pipeline_kwargs["device"] = -1

        self._tokenizer = None
        self._model = None
        self._pipe = None
        self.load()

    def load(self) -> None:
        # Idempotent: si déjà chargé, on ne refait rien
        if self._pipe is not None:
            return
        print(f"[Model] Loading model '{self.name}' for task '{self.task}'...")

        print("[Model] Checking tokenizer...")
        if "use_fast" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["use_fast"] = True
        tok = AutoTokenizer.from_pretrained(self.name, **self.tokenizer_kwargs)
        _ensure_pad_token(tok)

        print("[Model] Settling model kwargs...")
        self.model_kwargs.setdefault("low_cpu_mem_usage", True)
        self.model_kwargs.setdefault("use_safetensors", True)

        print("[Model] Downloading model...")
        mdl = AutoModelForCausalLM.from_pretrained(self.name, **self.model_kwargs)

        print("[Model] Defining pipeline...")
        self._pipe = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            **self.pipeline_kwargs,
        )
        print("[Model] Load completed.")

    def run(
        self,
        dataset,
        prompt: str | None,
        *,
        output_col: str = "Output",
        **gen_overrides,
    ):
        # --- Validations basiques ---
        if not hasattr(dataset, "data"):
            raise ValueError("[Model] No attribute 'data' found on dataset.")

        if not isinstance(dataset.data, pd.DataFrame):
            raise TypeError(
                "[Model] The 'data' attribute of dataset must be a pandas DataFrame."
            )

        if not hasattr(dataset, "field"):
            raise ValueError("[Model] The dataset must define the 'field' attribute.")

        if dataset.field not in dataset.data.columns:
            raise KeyError(f"[Model] Column '{dataset.field}' not found.")

        # --- Copie de travail ---
        print("[Model] Copying dataset...")
        df = dataset.data.copy()

        print(f"[Model] Keeping column '{dataset.field}' as text source.")
        texts = df[dataset.field].astype(str).tolist()

        # Paramètres d'inférence: overrides > configuration du modèle
        infer_kwargs = {**self.generation_kwargs, **gen_overrides}

        # Prompt effectif: argument > attribut du modèle > chaîne vide
        effective_prompt = prompt if prompt is not None else getattr(self, "prompt", "")

        outs = []
        for i, txt in enumerate(texts, start=1):
            print(f"\n===== Note {i}/{len(texts)} =====")
            print("[Model] Defining the prompt...")
            inp = f"{effective_prompt}\n\n{txt}".strip()

            print("[Model] Generating...")
            result = self._pipe(inp, **infer_kwargs)

            print("[Model] Checking result format...")
            if isinstance(result, list) and result:
                outs.append(result[0].get("generated_text", ""))
                print("[Model] OK result is a non-empty list.")
            elif isinstance(result, dict):
                outs.append(result.get("generated_text", ""))
                print("[Model] OK result is a dict.")
            else:
                outs.append(str(result))
                print(
                    "[Model] Warning: result is not a list or dict, storing str(result)."
                )

        print(f"[Model] Generated {len(outs)} outputs.")
        print("[Model] Determining output column name...")

        # Garantir l'unicité du nom de colonne demandé (défaut: "Output")
        base, unique_name = output_col, output_col
        i = 1
        while unique_name in df.columns:
            unique_name = f"{base}_{i}"
            i += 1

        df[unique_name] = outs
        print(f"[Model] Final output column name: {unique_name}")
        print(f"[Model] out is: {outs}")

        # --- Post-traitement sûr : extraction JSON -> colonnes structurées ---
        parsed_series = df[unique_name].apply(_extract_json_block)
        df["Symptoms"] = parsed_series.apply(lambda d: d.get("Symptoms", []))
        df["MedicalConclusion"] = parsed_series.apply(
            lambda d: d.get("MedicalConclusion", [])
        )
        df["Treatments"] = parsed_series.apply(lambda d: d.get("Treatments", []))
        df["Summary"] = parsed_series.apply(lambda d: d.get("Summary", ""))

        # --- Retourner un dataset du même type, sans side-effect sur l'original ---
        result_ds = copy.copy(dataset)
        result_ds.data = df
        result_ds.last_output_col = unique_name
        return result_ds
