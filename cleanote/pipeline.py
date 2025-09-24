# cleanote/pipeline.py
from typing import List, Optional, Tuple
from .types import Doc, Context, Report
from .data_downloader import DataDownloader
from .model_loader import ModelLoader
from .homogeniser import Homogeniser
from .verifier import Verifier


class Pipeline:
    def __init__(
        self,
        downloader: DataDownloader,
        homogeniser: Homogeniser,
        verifier: Optional[Verifier] = None,
        models: Optional[ModelLoader] = None,
    ) -> None:
        self.downloader = downloader
        self.homogeniser = homogeniser
        self.verifier = verifier
        self.models = models

    def run(self, ctx: Context) -> Tuple[List[Doc], List[Report]]:
        print("[Pipeline] Starting pipeline execution...")

        if self.models:
            print("[Pipeline] Preloading models...")
            self.models.preload(ctx)
            print("[Pipeline] Models loaded into context.")

        docs_out: List[Doc] = []
        reports: List[Report] = []

        print("[Pipeline] Fetching documents from DataDownloader...")
        for doc in self.downloader.fetch(ctx):
            print(f"[Pipeline] Processing document {doc.id}")

            d = self.homogeniser.run(doc, ctx)
            print(f"[Pipeline] Homogenisation done for {doc.id}")

            if self.verifier:
                d, rep = self.verifier.run(d, ctx)
                reports.append(rep)
                print(
                    f"[Pipeline] Verification done for {doc.id} with {len(rep.issues)} issues"
                )

            docs_out.append(d)

        print(f"[Pipeline] Finished. {len(docs_out)} documents processed.")
        return docs_out, reports
