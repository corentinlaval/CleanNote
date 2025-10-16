# tests/test_model_extra.py
import pandas as pd
import pytest

from cleanote.model import (
    Model,
    _split_kwargs_simple,
    _normalize_dtypes,
    _extract_json_block,
)


# ---------- Doubles / utilitaires ----------
class FakeTokenizer:
    def __init__(self, pad_token_id=None, eos_token_id=1, **kwargs):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.eos_token = "<eos>"
        self.pad_token = None
        self.kwargs = kwargs


class FakeCausalModel:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs


class PipelineRecorder:
    def __init__(self, task, model, tokenizer, **kwargs):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.calls = []

    def __call__(self, inputs, **infer_kwargs):
        self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
        return [{"generated_text": "GEN_OUT"}]


@pytest.fixture
def patch_transformers(monkeypatch):
    created = {}

    def fake_auto_tokenizer_from_pretrained(name, **kwargs):
        created["tokenizer_called_with"] = {"name": name, **kwargs}
        return FakeTokenizer(pad_token_id=None, eos_token_id=1, **kwargs)

    def fake_causal_from_pretrained(name, **kwargs):
        created["causal_model_called_with"] = {"name": name, **kwargs}
        return FakeCausalModel(name, **kwargs)

    def fake_pipeline(task, model, tokenizer, **kwargs):
        created["pipeline_called_with"] = {"task": task, "kwargs": kwargs}
        return PipelineRecorder(task, model, tokenizer, **kwargs)

    monkeypatch.setattr(
        "cleanote.model.AutoTokenizer.from_pretrained",
        fake_auto_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "cleanote.model.AutoModelForCausalLM.from_pretrained",
        fake_causal_from_pretrained,
    )
    monkeypatch.setattr("cleanote.model.pipeline", fake_pipeline)
    return created


class FakeDataset:
    def __init__(self, df: pd.DataFrame, field: str = "full_note"):
        self.data = df
        self.field = field


# ---------- Tests unitaires des helpers ----------
def test_split_kwargs_simple_routing_and_normalization():
    kw = dict(
        # prefixes
        model_revision="main",
        tokenizer_use_fast=False,
        # generation key (e.g. from GenerationConfig)
        max_new_tokens=32,
        # pipeline keys
        batch_size=8,
        device_map="auto",
        # dtype normalization targets
        dtype="float16",  # pipeline dtype -> torch_dtype
        model_dtype="bfloat16",  # model dtype -> torch_dtype
        # unknowns should fall back to generation_kwargs
        unknown_flag=True,
    )

    p_kw, g_kw, m_kw, t_kw = _split_kwargs_simple(kw)

    # prefixes routed
    assert t_kw == {"use_fast": False}
    assert m_kw["revision"] == "main"

    # generation keys (known + unknown fallback)
    assert g_kw["max_new_tokens"] == 32
    assert g_kw["unknown_flag"] is True

    # pipeline keys
    assert p_kw["batch_size"] == 8
    assert p_kw["device_map"] == "auto"

    # dtype normalized
    assert p_kw.get("torch_dtype") == "float16" and "dtype" not in p_kw
    assert m_kw.get("torch_dtype") == "bfloat16" and "dtype" not in m_kw


def test_normalize_dtypes_no_override_if_torch_dtype_present():
    d = {"dtype": "float16", "torch_dtype": "bfloat16"}
    # ne doit rien changer si torch_dtype déjà là
    _normalize_dtypes(d)
    assert d["torch_dtype"] == "bfloat16"
    assert d["dtype"] == "float16"  # inchangé (pas pop puisque torch_dtype existe)


def test_extract_json_block_valid_and_json_error():
    valid = 'prefix {"a": 1, "b": [2]} suffix'
    assert _extract_json_block(valid) == {"a": 1, "b": [2]}

    # JSON mal formé -> JSONDecodeError capturé -> {}
    invalid = "hello {not: valid json} world"
    assert _extract_json_block(invalid) == {}

    # Sans accolade -> {}
    none = "no json here"
    assert _extract_json_block(none) == {}


# ---------- Tests complémentaires de Model.run ----------
def test_run_injects_prompt_with_blank_line(monkeypatch, patch_transformers):
    # Enregistre l'input exact passé au pipeline
    df = pd.DataFrame({"full_note": ["hello"]})
    ds = FakeDataset(df)

    class CapturingPipe(PipelineRecorder):
        def __call__(self, inputs, **infer_kwargs):
            self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
            return [{"generated_text": "OK"}]

    monkeypatch.setattr(
        "cleanote.model.pipeline",
        lambda *a, **kw: CapturingPipe(*a, **kw),
    )

    m = Model(name="x/y", task="text-generation", tokenizer_use_fast=True)
    out = m.run(ds, prompt="PROMPT")
    assert out.data["Output"].iloc[0] == "OK"

    # Vérifie le format exact "PROMPT\n\n<texte>"
    call_inp = m._pipe.calls[-1]["inputs"]
    assert call_inp == "PROMPT\n\nhello"


def test_run_empty_list_response(monkeypatch, patch_transformers):
    # Couvre le chemin: isinstance(result, list) mais list vide -> fallback "str(result)"
    class EmptyListPipe:
        def __init__(self, *a, **kw):
            self.calls = []

        def __call__(self, inputs, **infer_kwargs):
            self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
            return []  # liste vide

    monkeypatch.setattr("cleanote.model.pipeline", lambda *a, **kw: EmptyListPipe())

    df = pd.DataFrame({"full_note": ["x"]})
    ds = FakeDataset(df)
    m = Model(name="repo/model", task="text-generation")

    out = m.run(ds, "p")
    # str([]) -> "[]"
    assert out.data["Output"].iloc[0] == "[]"


def test_tokenizer_and_model_prefix_kwargs_routed(monkeypatch):
    # Vérifie que tokenizer_* et model_* atteignent bien les from_pretrained
    tok_seen = {}
    mdl_seen = {}

    def fake_tok_from_pretrained(name, **kw):
        tok_seen.update(kw)
        return FakeTokenizer()

    def fake_mdl_from_pretrained(name, **kw):
        mdl_seen.update(kw)
        return FakeCausalModel(name, **kw)

    class _P:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            return [{"generated_text": "ok"}]

    monkeypatch.setattr(
        "cleanote.model.AutoTokenizer.from_pretrained", fake_tok_from_pretrained
    )
    monkeypatch.setattr(
        "cleanote.model.AutoModelForCausalLM.from_pretrained", fake_mdl_from_pretrained
    )
    monkeypatch.setattr("cleanote.model.pipeline", lambda *a, **kw: _P())

    _ = Model(
        name="x/y",
        task="text-generation",
        tokenizer_revision="tokrev",
        model_revision="mdlrev",
    )

    assert tok_seen.get("revision") == "tokrev"
    assert mdl_seen.get("revision") == "mdlrev"


def test_pipeline_keys_routed_to_pipeline_kwargs(patch_transformers):
    # batch_size & framework doivent se retrouver dans pipeline kwargs
    _ = Model(
        name="x/y",
        task="text-generation",
        batch_size=16,
        framework="pt",
    )
    pkw = patch_transformers["pipeline_called_with"]["kwargs"]
    assert pkw.get("batch_size") == 16
    assert pkw.get("framework") == "pt"


def test_multiple_output_col_collision_produces_incremented_suffix(patch_transformers):
    df = pd.DataFrame({"full_note": ["a"], "Output": ["e1"], "Output_1": ["e2"]})
    ds = FakeDataset(df)
    m = Model(name="x/y", task="text-generation")
    out = m.run(ds, "p")
    assert "Output_2" in out.data.columns
