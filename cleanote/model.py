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

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    num_beams: Optional[int] = None
    do_sample: bool = True
    # Add more fields if you need them (repetition_penalty, length_penalty, etc.)

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
            # If beams > 1, many prefer deterministic decoding:
            if "do_sample" in out and out["do_sample"] and self.num_beams > 1:
                # Keep as-is; you can set do_sample=False if you want beam search only.
                pass
        return out


class Model:
    """
    Generic HF model wrapper that can be instantiated with only a Hugging Face repo id.
    Provides initialize(), transform(), and transform_many() for integration with Homogeniser.

    Logs are concise and professional, in English.
    """

    def __init__(
        self,
        model_name: str,
        *,
        task: str = "text-generation",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        device_map: Optional[str] = "auto",  # uses Accelerate if available
        torch_dtype: Optional[
            str
        ] = None,  # e.g., "auto" or "bfloat16" (string to avoid torch import here)
        prompt_template: str = "{text}",  # simple passthrough by default
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

    # ---- Lifecycle -----------------------------------------------------------

    def initialize(self, ctx: Optional["Context"] = None) -> None:
        """Load the HF pipeline."""
        print(
            f"[Model] Initializing HF pipeline (model='{self.name}', task='{self.task}', revision={self.revision})..."
        )
        pipe_kwargs: Dict[str, Any] = {
            "task": self.task,
            "model": self.name,
            "trust_remote_code": self.trust_remote_code,
        }
        # Optional params only if provided
        if self.revision is not None:
            pipe_kwargs["revision"] = self.revision
        if self.device_map is not None:
            pipe_kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            pipe_kwargs["torch_dtype"] = (
                self.torch_dtype
            )  # string accepted by HF; or pass actual dtype if you prefer

        # Tokenizer will default to the same repo unless remote code needs something special.
        self._pipe = pipeline(**pipe_kwargs)
        print("[Model] Initialization completed.")

    # ---- Helpers -------------------------------------------------------------

    def _format_prompt(self, text: str) -> str:
        try:
            return self.prompt_template.format(text=text)
        except Exception:
            # Fallback: avoid crashing on malformed templates
            return text

    def _extract_text(self, raw_output: Any) -> str:
        """
        Normalize HF pipeline outputs to plain text.
        Handles common keys across tasks.
        """
        # Most pipelines return a list of dicts
        if isinstance(raw_output, list) and raw_output:
            item = raw_output[0]
            if isinstance(item, dict):
                # Common keys by task
                for key in (
                    "generated_text",
                    "summary_text",
                    "translation_text",
                    "answer",
                    "text",
                ):
                    if key in item and isinstance(item[key], str):
                        return item[key]
        # Fallback: string or repr
        if isinstance(raw_output, str):
            return raw_output
        return str(raw_output)

    # ---- Inference API -------------------------------------------------------

    def transform(self, doc, ctx) -> Any:
        """
        Synchronous single-doc call. Expects `doc` with a `text` attribute.
        Returns a new Doc with updated text, preserving id and adding model metadata if desired.
        """
        print(f"[Model] Transforming document {getattr(doc, 'id', '<no-id>')}...")
        if self._pipe is None:
            raise RuntimeError(
                "Model is not initialized. Call initialize() before transform()."
            )

        input_text = getattr(doc, "text", None)
        if not isinstance(input_text, str):
            raise TypeError("doc.text must be a string.")

        prompt = self._format_prompt(input_text)
        print(f"[Model] Prompt: {prompt[:50]}...")
        outputs = self._pipe(prompt, **self._gen.to_kwargs())
        print(f"[Model] Raw output: {outputs}")
        out_text = self._extract_text(outputs)
        print(f"[Model] Extracted text: {out_text[:50]}...")

        # Rebuild a Doc of the same type as input
        DocType = type(doc)
        meta = getattr(doc, "meta", {}) or {}
        meta_update = dict(meta)
        meta_update["model"] = {
            "name": self.name,
            "task": self.task,
            "revision": self.revision,
        }
        # Create new Doc (assumes signature: Doc(id=..., text=..., meta=...))
        try:
            return DocType(id=doc.id, text=out_text, meta=meta_update)
        except TypeError:
            # If your Doc uses `content` instead of `text`, fallback:
            return DocType(id=doc.id, content=out_text, meta=meta_update)

    def transform_many(self, docs: Iterable[Any], ctx) -> List[Any]:
        """
        Batched call. Accepts any iterable of Doc-like objects with a `text` attribute.
        Uses the HF pipeline with batched inputs for efficiency.
        """
        if self._pipe is None:
            raise RuntimeError(
                "Model is not initialized. Call initialize() before transform_many()."
            )

        docs_list = list(docs)
        if not docs_list:
            return []

        prompts = [self._format_prompt(getattr(d, "text", "")) for d in docs_list]
        # Batched inference; transformers pipelines accept a list of inputs
        outputs = self._pipe(prompts, **self._gen.to_kwargs())

        # Normalize outputs to a list (HF pipelines may return list-of-lists depending on params)
        if outputs and isinstance(outputs, list) and not isinstance(outputs[0], dict):
            # e.g., [[{generated_text: ...}], [{...}], ...]
            flat = []
            for item in outputs:
                flat.append(self._extract_text(item))
            texts = flat
        else:
            texts = [self._extract_text(o) for o in outputs]

        if len(texts) != len(docs_list):
            # Conservative safeguard to avoid silent misalignment
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
