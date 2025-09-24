# cleanote/pipeline.py
from typing import List, Optional, Tuple
from .types import Doc, Context, Report
from .data_downloader import DataDownloader
from .model import Model
from .homogeniser import Homogeniser
from .verifier import Verifier


class Pipeline:
    def __init__(
        self,
        downloader: DataDownloader,
        homogeniser: Homogeniser,
        verifier: Verifier,
        homogeniser_model: Optional[Model] = None,
        verifier_model: Optional[Model] = None,
    ) -> None:
        self.downloader = downloader
        self.homogeniser = homogeniser
        self.verifier = verifier
        self.homogeniser_model = homogeniser_model
        self.verifier_model = verifier_model

    def run(self, ctx: Context) -> Tuple[List[Doc], List[Report]]:
        print("[Pipeline] Starting pipeline execution...")

        if self.homogeniser_model:
            print("[Pipeline] Preloading homogeniser model...")
            self.homogeniser_model.initialize(ctx)
            print("[Pipeline] Homogeniser model loaded into context.")

        if self.verifier_model:
            print("[Pipeline] Preloading verifier model...")
            self.verifier_model.initialize(ctx)
            print("[Pipeline] Verifier model loaded into context.")

        docs_out: List[Doc] = []
        reports: List[Report] = []

        print("[Pipeline] Fetching documents from DataDownloader...")
        for doc in self.downloader.fetch(ctx):
            print(f"[Pipeline] Processing document {doc.id}")

            d = self.homogeniser.run(self.homogeniser_model, doc, ctx)
            print(f"[Pipeline] Homogenisation done for {doc.id}")

            d = self.verifier.run(self.verifier_model, d, ctx)
            docs_out.append(d)

        print(f"[Pipeline] Finished. {len(docs_out)} documents processed.")
        return docs_out, reports
