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
        if self.models:
            self.models.preload(ctx)
        docs_out: List[Doc] = []
        reports: List[Report] = []
        for doc in self.downloader.fetch(ctx):
            d = self.homogeniser.run(doc, ctx)
            if self.verifier:
                d, rep = self.verifier.run(d, ctx)
                reports.append(rep)
            docs_out.append(d)
        return docs_out, reports
