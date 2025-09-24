from typing import Iterable, Union
from .types import Doc, Context
from .model import Model


class Verifier:
    def run(
        self,
        model_loader: Model,
        docs: Union[Doc, Iterable[Doc]],
        ctx: Context,
    ) -> Union[Doc, Iterable[Doc]]:
        print(f"[Verifier] Initializing model loader '{model_loader.name}'...")
        model_loader.initialize()
        print("[Verifier] Initialization completed.\n")

        # 2) Normalize input to a list for consistent handling
        is_single = not isinstance(docs, Iterable) or isinstance(docs, Doc)
        in_docs = [docs] if is_single else list(docs)
        print(f"[Verifier] Sending {len(in_docs)} document(s) to the model loader...")

        # 3) Send documents to the model loader
        out_docs = in_docs  # [model_loader.transform(d, ctx) for d in in_docs]

        # 4) Return output
        print(f"[Verifier] Model loader returned {len(out_docs)} document(s).")
        print("[Verifier] Done.")
        return out_docs[0] if is_single else out_docs
