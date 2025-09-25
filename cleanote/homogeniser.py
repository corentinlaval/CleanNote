from typing import Iterable, Union
from .types import Doc, Context
from .model import Model


class Homogeniser:
    def __init__(
        self,
        prompt_template: str = "Count the number of words in the following text:\n\n{text}",
    ) -> None:
        # You can change the prompt without touching the code
        self.prompt_template = prompt_template

    def run(
        self,
        model_loader: Model,
        docs: Union[Doc, Iterable[Doc]],
        ctx: Context,
    ) -> Union[Doc, Iterable[Doc]]:
        if model_loader is None:
            raise ValueError("Homogeniser requires a model_loader instance.")
        # IMPORTANT: do NOT initialize here; the Pipeline already does it.

        # 1) Normalize input
        is_single = isinstance(docs, Doc)
        in_docs = [docs] if is_single else list(docs)
        print(
            f"[Homogeniser] Sending {len(in_docs)} document(s) to the model loader..."
        )

        # 2) Build prompt and call model
        out_docs: list[Doc] = []
        for d in in_docs:
            prompt = self.prompt_template.format(text=d.text)
            print(f"[Homogeniser] Prompt for doc {d.id}: {prompt[:80]}...")
            tmp = Doc(id=d.id, text=prompt, meta=getattr(d, "meta", {}))
            out = model_loader.transform(tmp, ctx)
            out_docs.append(out)

        # 3) Return
        print(f"[Homogeniser] Model loader returned {len(out_docs)} document(s).")
        print("[Homogeniser] Done.")
        return out_docs[0] if is_single else out_docs
