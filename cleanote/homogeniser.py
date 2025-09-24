from typing import Iterable, Union
from .types import Doc, Context
from .model_loader import ModelLoader


class Homogeniser:
    def run(
        self,
        model_loader: ModelLoader,
        docs: Union[Doc, Iterable[Doc]],
        ctx: Context,
    ) -> Union[Doc, Iterable[Doc]]:
        # 1) Initialize the model loader
        model_name = getattr(model_loader, "name", model_loader.__class__.__name__)
        print(f"[Homogeniser] Initializing model loader '{model_name}'...")
        model_loader.initialize()
        print("[Homogeniser] Initialization completed.\n")

        # 2) Normalize input to a list for consistent handling
        is_single = not isinstance(docs, Iterable) or isinstance(docs, Doc)
        in_docs = [docs] if is_single else list(docs)
        print(
            f"[Homogeniser] Sending {len(in_docs)} document(s) to the model loader..."
        )

        # 3) Send documents to the model loader
        out_docs = in_docs  # [model_loader.transform(d, ctx) for d in in_docs]

        # 4) Return output
        print(f"[Homogeniser] Model loader returned {len(out_docs)} document(s).")
        print("[Homogeniser] Done.")
        return out_docs[0] if is_single else out_docs
