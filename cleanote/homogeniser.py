from typing import List, Protocol
from .types import Doc, Context


class _Step(Protocol):
    def run(self, doc: Doc, ctx: Context) -> Doc: ...


class Homogeniser:
    def __init__(self, steps: List[_Step]) -> None:
        self.steps = steps

    def run(self, doc: Doc, ctx: Context) -> Doc:
        out = doc
        for s in self.steps:
            out = s.run(out, ctx)
        return out
