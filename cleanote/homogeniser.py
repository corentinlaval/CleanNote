from typing import Iterable, Union
from .types import Doc, Context
from .model import Model


class Homogeniser:
    def run(
        self,
        model_loader: Model,
        docs: Union[Doc, Iterable[Doc]],
        ctx: Context,
    ) -> Union[Doc, Iterable[Doc]]:
        print(f"[Homogeniser] Initializing model loader '{model_loader.name}'...")
        model_loader.initialize()
        print("[Homogeniser] Initialization completed.\n")

        # 2) Normalize input to a list for consistent handling
        is_single = isinstance(docs, Doc)
        in_docs = [docs] if is_single else list(docs)
        print(
            f"[Homogeniser] Sending {len(in_docs)} document(s) to the model loader..."
        )

        # 3) Send documents to the model loader with a prompt
        out_docs = []
        for d in in_docs:
            # Construire un prompt très simple
            prompt = f"Count the number of words in the following text:\n\n{d.text}"
            print(f"[Homogeniser] Prompt for doc {d.id}: {prompt[:50]}...")
            # Transformer le Doc en un nouveau Doc via le modèle
            transformed = model_loader.transform(
                Doc(id=d.id, text=prompt, meta=d.meta), ctx
            )
            out_docs.append(transformed)

        # 4) Return output
        print(f"[Homogeniser] Model loader returned {len(out_docs)} document(s).")
        print("[Homogeniser] Done.")
        return out_docs[0] if is_single else out_docs
