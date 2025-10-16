# tests/test_model_all.py
import json
import pandas as pd
import pytest

from cleanote.model import (
    Model,
    _split_kwargs_simple,
    _normalize_dtypes,
    _extract_json_block,
)

# -------------------------- Doubles (fakes) --------------------------


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
    """Enregistre kwargs passés à pipeline() et les inférences."""

    def __init__(self, task, model, tokenizer, **kwargs):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.calls = []

    def __call__(self, inputs, **infer_kwargs):
        self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
        return [{"generated_text": "GEN_OUT"}]


# -------------------------- Fixtures de patch --------------------------


@pytest.fixture
def patch_transformers(monkeypatch):
    created = {}

    def fake_tok_from_pretrained(name, **kwargs):
        created["tok_called_with"] = {"name": name, **kwargs}
        # pad_token_id=None pour forcer _ensure_pad_token à copier eos->pad
        return FakeTokenizer(pad_token_id=None, eos_token_id=99, **kwargs)

    def fake_m_from_pretrained(name, **kwargs):
        created["model_called_with"] = {"name": name, **kwargs}
        return FakeCausalModel(name, **kwargs)

    def fake_pipeline(task, model, tokenizer, **kwargs):
        created["pipeline_called_with"] = {"task": task, "kwargs": kwargs}
        return PipelineRecorder(task, model, tokenizer, **kwargs)

    monkeypatch.setattr(
        "cleanote.model.AutoTokenizer.from_pretrained", fake_tok_from_pretrained
    )
    monkeypatch.setattr(
        "cleanote.model.AutoModelForCausalLM.from_pretrained", fake_m_from_pretrained
    )
    monkeypatch.setattr("cleanote.model.pipeline", fake_pipeline)
    return created


# -------------------------- Aide dataset --------------------------


class FakeDataset:
    def __init__(self, df: pd.DataFrame, field: str = "full_note"):
        self.data = df
        self.field = field


# -------------------------- Tests helpers --------------------------


def test_split_kwargs_simple_routing_and_normalization():
    kw = dict(
        # prefixes
        model_revision="main",
        tokenizer_use_fast=False,
        # generation (clé connue de GenerationConfig)
        max_new_tokens=32,
        # pipeline keys
        batch_size=8,
        device_map="auto",
        # dtype normalisation
        dtype="float16",  # -> pipeline_kwargs.torch_dtype
        model_dtype="bfloat16",  # -> model_kwargs.torch_dtype
        # inconnu -> generation_kwargs
        unknown_flag=True,
    )
    pkw, gkw, mkw, tkw = _split_kwargs_simple(kw)

    assert tkw == {"use_fast": False}
    assert mkw["revision"] == "main"
    assert gkw["max_new_tokens"] == 32 and gkw["unknown_flag"] is True
    assert pkw["batch_size"] == 8 and pkw["device_map"] == "auto"
    assert pkw.get("torch_dtype") == "float16" and "dtype" not in pkw
    assert mkw.get("torch_dtype") == "bfloat16" and "dtype" not in mkw


def test_normalize_dtypes_no_override_if_torch_dtype_present():
    d = {"dtype": "float16", "torch_dtype": "bfloat16"}
    _normalize_dtypes(d)
    # ne modifie pas torch_dtype existant, et ne pop pas dtype
    assert d["torch_dtype"] == "bfloat16"
    assert d["dtype"] == "float16"


def test_extract_json_block_variants():
    assert _extract_json_block('x {"a": 1, "b": [2]} y') == {"a": 1, "b": [2]}
    # JSON invalide -> {}
    assert _extract_json_block("bad {not: valid json} text") == {}
    # pas de JSON -> {}
    assert _extract_json_block("nothing here") == {}


# -------------------------- Tests Model.load --------------------------


def test_load_sets_defaults_and_is_idempotent(patch_transformers, capsys):
    m = Model(name="repo/model", task="text-generation")
    out = capsys.readouterr().out
    assert "Loading model 'repo/model' for task 'text-generation'..." in out
    assert "Load completed." in out

    # tokenizer.use_fast par défaut True
    assert patch_transformers["tok_called_with"]["use_fast"] is True
    # device par défaut -1 et return_full_text False pour pipeline
    pkw = patch_transformers["pipeline_called_with"]["kwargs"]
    assert pkw.get("device", -2) == -1
    assert pkw.get("return_full_text") is False
    # model kwargs par défaut
    assert patch_transformers["model_called_with"]["low_cpu_mem_usage"] is True
    assert patch_transformers["model_called_with"]["use_safetensors"] is True

    # idempotence
    first = m._pipe
    m.load()
    assert m._pipe is first


def test_tokenizer_and_model_prefix_kwargs_routed(monkeypatch):
    seen_tok, seen_m = {}, {}

    def fake_tok(name, **kw):
        seen_tok.update(kw)
        return FakeTokenizer()

    def fake_m(name, **kw):
        seen_m.update(kw)
        return FakeCausalModel(name, **kw)

    class _P:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            return [{"generated_text": "ok"}]

    monkeypatch.setattr("cleanote.model.AutoTokenizer.from_pretrained", fake_tok)
    monkeypatch.setattr("cleanote.model.AutoModelForCausalLM.from_pretrained", fake_m)
    monkeypatch.setattr("cleanote.model.pipeline", lambda *a, **kw: _P())

    _ = Model(
        name="n/h",
        task="text-generation",
        tokenizer_revision="tokrev",
        model_revision="mdlrev",
        tokenizer_use_fast=False,  # override explicit
    )
    assert seen_tok.get("revision") == "tokrev"
    assert seen_tok.get("use_fast") is False
    assert seen_m.get("revision") == "mdlrev"


# -------------------------- Tests Model.run (branches) --------------------------


def test_run_happy_path_adds_default_Output_and_keeps_input(patch_transformers):
    df = pd.DataFrame({"full_note": ["hello", "world"]})
    ds = FakeDataset(df)
    m = Model(name="repo/model", task="text-generation", prompt="PROMPT")
    out = m.run(ds)
    assert "Output" in out.data.columns
    assert list(out.data["Output"]) == ["GEN_OUT", "GEN_OUT"]
    # l’input est préservé
    assert list(out.data[ds.field]) == ["hello", "world"]
    # pointeur utile
    assert out.last_output_col == "Output"


def test_run_respects_explicit_output_col_and_collision(patch_transformers):
    df = pd.DataFrame({"full_note": ["x"], "text_out": ["exists already"]})
    ds = FakeDataset(df)
    m = Model(name="repo/model", task="text-generation", prompt="p")
    out = m.run(ds, output_col="text_out")
    # collision -> suffix _1
    assert "text_out_1" in out.data.columns
    assert list(out.data["text_out_1"]) == ["GEN_OUT"]


def test_run_uses_self_prompt_and_passes_overrides(monkeypatch):
    class Rec(PipelineRecorder):
        def __call__(self, inputs, **infer_kwargs):
            self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
            return [{"generated_text": "OK"}]

    monkeypatch.setattr(
        "cleanote.model.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        "cleanote.model.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: FakeCausalModel(a[0], **k),
    )
    monkeypatch.setattr("cleanote.model.pipeline", lambda *a, **kw: Rec(*a, **kw))

    df = pd.DataFrame({"full_note": ["note"]})
    ds = FakeDataset(df)
    m = Model(
        name="repo/model",
        task="text-generation",
        prompt="PROMPT",
        max_new_tokens=128,
        temperature=0.5,
    )
    out = m.run(ds, max_new_tokens=3, temperature=0.0)  # overrides
    assert out.data["Output"].iloc[0] == "OK"

    # format exact de l’input : "PROMPT\n\n<texte>"
    last = m._pipe.calls[-1]
    assert last["inputs"] == "PROMPT\n\nnote"
    # overrides priment
    assert last["infer_kwargs"]["max_new_tokens"] == 3
    assert last["infer_kwargs"]["temperature"] == 0.0


def test_run_handles_dict_and_str_responses(monkeypatch):
    # dict
    class DictPipe:
        def __init__(self, *a, **kw):
            self.calls = []

        def __call__(self, inputs, **infer_kwargs):
            self.calls.append(1)
            return {"generated_text": "D"}

    # str (ni list, ni dict)
    class StrPipe:
        def __init__(self, *a, **kw):
            self.calls = []

        def __call__(self, inputs, **infer_kwargs):
            self.calls.append(1)
            return "S"

    # patch tokenizer/model
    monkeypatch.setattr(
        "cleanote.model.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        "cleanote.model.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: FakeCausalModel(a[0], **k),
    )

    # dict case
    monkeypatch.setattr("cleanote.model.pipeline", lambda *a, **kw: DictPipe(*a, **kw))
    m = Model(name="repo/model", task="text-generation", prompt="")
    out = m.run(FakeDataset(pd.DataFrame({"full_note": ["a"]})))
    assert out.data["Output"].iloc[0] == "D"

    # str case
    monkeypatch.setattr("cleanote.model.pipeline", lambda *a, **kw: StrPipe(*a, **kw))
    m2 = Model(name="repo/model", task="text-generation", prompt="")
    out2 = m2.run(FakeDataset(pd.DataFrame({"full_note": ["a"]})))
    assert out2.data["Output"].iloc[0] == "S"


def test_run_list_empty_falls_back_to_str(monkeypatch):
    class EmptyListPipe:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, inputs, **infer_kwargs):
            return []

    monkeypatch.setattr(
        "cleanote.model.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        "cleanote.model.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: FakeCausalModel(a[0], **k),
    )
    monkeypatch.setattr("cleanote.model.pipeline", lambda *a, **kw: EmptyListPipe())

    m = Model(name="repo/model", task="text-generation", prompt="p")
    out = m.run(FakeDataset(pd.DataFrame({"full_note": ["z"]})))
    # str([]) == "[]"
    assert out.data["Output"].iloc[0] == "[]"


# -------------------------- Tests erreurs d’entrées --------------------------


def test_errors_dataset_without_dataframe(patch_transformers):
    class BadDs:
        def __init__(self):
            self.data = {"full_note": ["x"]}  # pas un DataFrame
            self.field = "full_note"

    m = Model(name="x/y", task="text-generation", prompt="")
    with pytest.raises(TypeError):
        m.run(BadDs())


def test_errors_missing_field_attribute(patch_transformers):
    class BadDs:
        def __init__(self, df):
            self.data = df

    df = pd.DataFrame({"full_note": ["x"]})
    m = Model(name="x/y", task="text-generation", prompt="")
    with pytest.raises(ValueError):
        m.run(BadDs(df))


def test_errors_missing_column(patch_transformers):
    class Ds:
        def __init__(self, df):
            self.data = df
            self.field = "full_note"

    df = pd.DataFrame({"other": ["x"]})
    m = Model(name="x/y", task="text-generation", prompt="")
    with pytest.raises(KeyError):
        m.run(Ds(df))


# -------------------------- Tests post-processing JSON → colonnes --------------------------


def test_postprocessing_extracts_json_to_columns(monkeypatch):
    class JsonPipe:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            payload = {
                "Symptoms": ["A", "B"],
                "MedicalConclusion": ["C"],
                "Treatments": ["T1"],
                "Summary": "S",
            }
            return [{"generated_text": f"before {json.dumps(payload)} after"}]

    monkeypatch.setattr(
        "cleanote.model.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        "cleanote.model.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: FakeCausalModel(a[0], **k),
    )
    monkeypatch.setattr("cleanote.model.pipeline", lambda *a, **kw: JsonPipe())

    df = pd.DataFrame({"full_note": ["n1", "n2"]})
    ds = FakeDataset(df)
    m = Model(name="repo/model", task="text-generation", prompt="")
    out = m.run(ds)

    assert "Output" in out.data.columns
    assert out.data["Symptoms"].iloc[0] == ["A", "B"]
    assert out.data["MedicalConclusion"].iloc[0] == ["C"]
    assert out.data["Treatments"].iloc[0] == ["T1"]
    assert out.data["Summary"].iloc[0] == "S"


def test_postprocessing_defaults_when_no_json(monkeypatch):
    class PlainPipe:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            return [{"generated_text": "no json here"}]

    monkeypatch.setattr(
        "cleanote.model.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        "cleanote.model.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: FakeCausalModel(a[0], **k),
    )
    monkeypatch.setattr("cleanote.model.pipeline", lambda *a, **kw: PlainPipe())

    df = pd.DataFrame({"full_note": ["n"]})
    ds = FakeDataset(df)
    m = Model(name="repo/model", task="text-generation", prompt="")
    out = m.run(ds)

    assert out.data["Symptoms"].iloc[0] == []
    assert out.data["MedicalConclusion"].iloc[0] == []
    assert out.data["Treatments"].iloc[0] == []
    assert out.data["Summary"].iloc[0] == ""
