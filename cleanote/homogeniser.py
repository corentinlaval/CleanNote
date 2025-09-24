# cleanote/homogeniser.py
from typing import List, Protocol, Callable, Optional
from .types import Doc, Context


class _Step(Protocol):
    def run(self, doc: Doc, ctx: Context) -> Doc: ...


class Homogeniser:
    """
    Applies a list of steps sequentially to a Doc.
    - Keeps your original API.
    - Optional verbose logs.
    - Optional on_step callback for tracing/metrics.
    """

    def __init__(
        self,
        steps: List[_Step],
        *,
        verbose: bool = False,
        on_step: Optional[Callable[[str, Doc, Doc, Context], None]] = None,
    ) -> None:
        self.steps = steps
        self.verbose = verbose
        self.on_step = on_step

    def run(self, doc: Doc, ctx: Context) -> Doc:
        out = doc
        if self.verbose:
            print(f"[Homogeniser] Start doc {doc.id}")
        for s in self.steps:
            before = out
            out = s.run(out, ctx)
            if self.verbose:
                print(
                    f"  [Step] {s.__class__.__name__}: "
                    f"{(before.text[:60] if before.text else '')!r} -> "
                    f"{(out.text[:60] if out.text else '')!r}"
                )
            if self.on_step:
                self.on_step(s.__class__.__name__, before, out, ctx)
        if self.verbose:
            print(f"[Homogeniser] Done doc {doc.id}")
        return out
