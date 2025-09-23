from typing import List, Callable
from .types import Doc, Context, Issue, Report

Check = Callable[[Doc, Context], List[Issue]]


class Verifier:
    def __init__(self, checks: List[Check]) -> None:
        self.checks = checks

    def run(self, doc: Doc, ctx: Context):
        issues: List[Issue] = []
        for check in self.checks:
            issues.extend(check(doc, ctx))
        return doc, Report(doc_id=doc.id, issues=issues)
