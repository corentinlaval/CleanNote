# tests/test_pipeline_apply_orchestration.py
import json
import pandas as pd
from cleanote.pipeline import Pipeline


class _FakeDataset:
    def __init__(self, df, field="full_note"):
        self.data = df
        self.field = field


class _FakeModelH:
    def __init__(self, prompt=None):
        self.prompt = prompt

    def run(self, dataset, output_col="full_note__h", **_):
        out = _FakeDataset(dataset.data.copy(), dataset.field)
        out.data[output_col] = json.dumps(
            {"Symptoms": [], "MedicalConclusion": [], "Treatments": [], "Summary": ""}
        )
        return out


def test_apply_calls_steps_in_order(monkeypatch, capsys):
    df = pd.DataFrame({"full_note": ["x", "y"]})
    p = Pipeline(_FakeDataset(df), _FakeModelH())
    calls = []

    def fake_homogenize(self):
        calls.append("homogenize")
        out_h_col = f"{self.dataset.field}__h"
        self.dataset_h = self.model_h.run(self.dataset, output_col=out_h_col)

    def make_step(name):
        def _f(self):
            calls.append(name)

        return _f

    monkeypatch.setattr(Pipeline, "homogenize", fake_homogenize, raising=True)
    monkeypatch.setattr(
        Pipeline, "verify_QuickUMLS", make_step("verify_QuickUMLS"), raising=True
    )
    monkeypatch.setattr(Pipeline, "verify_NLI", make_step("verify_NLI"), raising=True)
    monkeypatch.setattr(
        Pipeline,
        "verify_UMLS_summary_vs_source",
        make_step("verify_UMLS_summary_vs_source"),
        raising=True,
    )

    out_ds = p.apply()

    assert calls == [
        "homogenize",
        "verify_QuickUMLS",
        "verify_NLI",
        "verify_UMLS_summary_vs_source",
    ]
    assert out_ds is p.dataset_h

    stdout = capsys.readouterr().out
    assert "[Pipeline] Starting pipeline..." in stdout
    assert "[Pipeline] Pipeline completed." in stdout
    assert f"{p.dataset.field}__h" in p.dataset_h.data.columns
