from cleanote.types import Doc, Context
from cleanote.homogeniser import Homogeniser


# micro-étape de démo (si tu n’en as pas encore dans ton fichier)
class NormalizeWhitespace:
    def __init__(self, keep_double_newlines=True):
        self.keep = keep_double_newlines

    def run(self, doc, ctx):
        import re

        t = doc.text
        t = re.sub(r"[ \t]+", " ", t)
        if self.keep:
            t = re.sub(r"\n{3,}", "\n\n", t)
        else:
            t = re.sub(r"\s+", " ", t)
        return doc.copy(update={"text": t})


def test_normalize_whitespace_unit():
    doc = Doc(id="1", text="a   b\t\tc")
    out = NormalizeWhitespace(keep_double_newlines=False).run(
        doc, Context(run_id="t", params={})
    )
    assert out.text == "a b c"


def test_homogeniser_sequence():
    h = Homogeniser([NormalizeWhitespace()])
    doc = Doc(id="1", text="a   b")
    out = h.run(doc, Context(run_id="t", params={}))
    assert out.text == "a b"
