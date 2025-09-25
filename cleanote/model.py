# cleanote/model.py
from __future__ import annotations

from typing import Iterable, List, Optional, Dict, Any
from dataclasses import dataclass
from .types import Context

try:
    from transformers import pipeline  # type: ignore
except Exception as e:
    raise RuntimeError(
        "The 'transformers' package is required to use Model. "
        "Install it with: pip install transformers accelerate torch"
    ) from e


@dataclass
class GenerationConfig:
    """Default generation parameters (override per your use-case)."""

    max_new_tokens: int = 32  # plus court par défaut => plus rapide
    temperature: float = 0.0  # déterministe pour tests
    top_p: float = 1.0
    top_k: Optional[int] = None
    num_beams: Optional[int] = None
    do_sample: bool = False  # pas d'échantillonnage par défaut

    def to_kwargs(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }
        if self.top_k is not None:
            out["top_k"] = self.top_k
        if self.num_beams is not None:
            out["num_beams"] = self.num_beams
        return out


class Model:
    """
    Generic HF model wrapper that can be instantiated with only a Hugging Face repo id.
    Provides initialize(), transform(), and transform_many().

    Logs are concise and professional, in English.
    """

    def __init__(
        self,
        model_name: str,
        *,
        task: str = "text-generation",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[str] = None,
        prompt_template: str = "{text}",
        generation: Optional[GenerationConfig] = None,
        batch_size: int = 8,
    ) -> None:
        self.name = model_name
        self.task = task
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.prompt_template = prompt_template
        self.batch_size = batch_size

        self._pipe = None
        self._gen = generation or GenerationConfig()
        self._tokenizer = None
        self._model_max_length = 1024  # mis à jour après init

    # ---- Lifecycle -----------------------------------------------------------

    def initialize(self, ctx: Optional[Context] = None) -> None:
        """Load the HF pipeline. `ctx` is accepted for compatibility and ignored."""
        print(
            f"[Model] Initializing HF pipeline (model='{self.name}', task='{self.task}', revision={self.revision})..."
        )
        pipe_kwargs: Dict[str, Any] = {
            "task": self.task,
            "model": self.name,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.revision is not None:
            pipe_kwargs["revision"] = self.revision
        if self.device_map is not None:
            pipe_kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            pipe_kwargs["torch_dtype"] = self.torch_dtype

        self._pipe = pipeline(**pipe_kwargs)
        # capture tokenizer et longueur max
        self._tokenizer = getattr(self._pipe, "tokenizer", None)
        self._model_max_length = getattr(self._tokenizer, "model_max_length", 1024)
        print("[Model] Initialization completed.")

    # Alias de compatibilité éventuel
    def preload(self, ctx: Optional[Context] = None) -> None:
        self.initialize(ctx)

    # ---- Helpers -------------------------------------------------------------

    def _format_prompt(self, text: str) -> str:
        try:
            return self.prompt_template.format(text=text)
        except Exception:
            return text

    def _truncate_for_model(self, prompt: str) -> str:
        """Truncate prompt to respect model context window, reserving space for generation."""
        if self._tokenizer is None:
            return prompt
        reserve = max(4, int(self._gen.max_new_tokens))
        max_input = max(16, int(self._model_max_length) - reserve)
        enc = self._tokenizer(
            prompt,
            truncation=True,
            max_length=max_input,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = enc["input_ids"]
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def _extract_text(self, raw_output: Any) -> str:
        if isinstance(raw_output, list) and raw_output:
            item = raw_output[0]
            if isinstance(item, dict):
                for key in (
                    "generated_text",
                    "summary_text",
                    "translation_text",
                    "answer",
                    "text",
                ):
                    if key in item and isinstance(item[key], str):
                        return item[key]
        if isinstance(raw_output, str):
            return raw_output
        return str(raw_output)

    def _call_pipeline(self, inputs) -> Any:
        gen = self._gen.to_kwargs()
        # éviter l'écho complet sur text-generation
        if self.task == "text-generation":
            gen.setdefault("return_full_text", False)
        # filet de sécurité
        gen.setdefault("truncation", True)
        gen.setdefault("max_length", int(self._model_max_length))
        return self._pipe(inputs, **gen)

    # ---- Inference API -------------------------------------------------------

    def transform(self, doc, ctx) -> Any:
        print(f"[Model] Transforming document {getattr(doc, 'id', '<no-id>')}...")
        if self._pipe is None:
            raise RuntimeError(
                "Model is not initialized. Call initialize() before transform()."
            )

        input_text = getattr(doc, "text", None)
        if not isinstance(input_text, str):
            raise TypeError("doc.text must be a string.")

        prompt = self._format_prompt(input_text)
        prompt = self._truncate_for_model(prompt)
        outputs = self._call_pipeline(prompt)
        out_text = self._extract_text(outputs)

        DocType = type(doc)
        meta = getattr(doc, "meta", {}) or {}
        meta_update = dict(meta)
        meta_update["model"] = {
            "name": self.name,
            "task": self.task,
            "revision": self.revision,
        }
        try:
            return DocType(id=doc.id, text=out_text, meta=meta_update)
        except TypeError:
            return DocType(id=doc.id, content=out_text, meta=meta_update)

    def transform_many(self, docs: Iterable[Any], ctx) -> List[Any]:
        if self._pipe is None:
            raise RuntimeError(
                "Model is not initialized. Call initialize() before transform_many()."
            )

        docs_list = list(docs)
        if not docs_list:
            return []

        prompts = []
        for d in docs_list:
            base = getattr(d, "text", "")
            p = self._format_prompt(base)
            p = self._truncate_for_model(p)
            prompts.append(p)

        outputs = self._call_pipeline(prompts)
        if outputs and isinstance(outputs, list) and not isinstance(outputs[0], dict):
            texts = [self._extract_text(item) for item in outputs]
        else:
            texts = [self._extract_text(o) for o in outputs]

        if len(texts) != len(docs_list):
            raise RuntimeError(
                f"Model produced {len(texts)} outputs for {len(docs_list)} inputs."
            )

        out_docs: List[Any] = []
        for src, out_text in zip(docs_list, texts):
            DocType = type(src)
            meta = getattr(src, "meta", {}) or {}
            meta_update = dict(meta)
            meta_update["model"] = {
                "name": self.name,
                "task": self.task,
                "revision": self.revision,
            }
            try:
                out_docs.append(DocType(id=src.id, text=out_text, meta=meta_update))
            except TypeError:
                out_docs.append(DocType(id=src.id, content=out_text, meta=meta_update))
        return out_docs
