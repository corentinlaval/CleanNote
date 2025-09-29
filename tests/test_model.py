# tests/test_model.py
import pandas as pd
import pytest

# Adapte l'import si besoin
from cleanote.model import Model


# ---------- Doubles de test (mocks légers) ----------
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
    """Capture les kwargs passés à pipeline() et la dernière inférence."""

    def __init__(self, task, model, tokenizer, **kwargs):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.calls = []  # liste des inputs reçus

    def __call__(self, inputs, **infer_kwargs):
        # inputs est une str (une entrée par appel)
        self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
        # simul text-generation
        return [{"generated_text": "GEN_OUT"}]


# ---------- Fixtures de monkeypatch ----------
@pytest.fixture
def patch_transformers(monkeypatch):
    created = {}

    def fake_auto_tokenizer_from_pretrained(name, **kwargs):
        created["tokenizer_called_with"] = {"name": name, **kwargs}
        # Simule l'absence de pad_token_id pour tester le fallback
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


# ---------- Aides ----------
class FakeDataset:
    def __init__(self, df: pd.DataFrame, field: str = "full_note"):
        self.data = df
        self.field = field
        self.name = "dummy/ds"
        self.limit = len(df)


# ---------- Tests ----------
def test_text_generation_happy_path_df_adds_column_and_sets_defaults(
    patch_transformers, capsys
):
    df = pd.DataFrame({"index": [0, 1], "full_note": ["hello", "world"]})
    ds = FakeDataset(df)

    # both max_new_tokens and max_length -> le second doit être retiré
    m = Model(
        name="facebook/opt-350m",
        task="text-generation",
        max_new_tokens=8,
        max_length=999,  # doit être ignoré
        tokenizer_use_fast=True,
        pipeline_batch_size=4,  # route vers pipeline_kwargs
    )

    # prints « Loading... » + « Load completed. »
    out_txt = capsys.readouterr().out
    assert "Loading model 'facebook/opt-350m' for task 'text-generation'..." in out_txt
    assert "Load completed." in out_txt

    # vérifie que le tokenizer a bien reçu use_fast=True
    assert patch_transformers["tokenizer_called_with"]["use_fast"] is True

    # run => ajoute une colonne auto-nommée
    out_ds = m.run(ds, prompt="count words please")
    new_cols = [c for c in out_ds.data.columns if c.startswith(ds.field + "__")]
    assert len(new_cols) == 1
    col = new_cols[0]
    assert list(out_ds.data[col]) == ["GEN_OUT", "GEN_OUT"]
    # l'original est préservé
    assert list(out_ds.data[ds.field]) == ["hello", "world"]
    # garde un pointeur vers la dernière colonne
    assert out_ds.last_output_col == col

    # pipeline kwargs: device par défaut = -1 si non fourni
    assert patch_transformers["pipeline_called_with"]["kwargs"].get("device", -2) == -1

    # generation_kwargs: max_length doit avoir été retiré si max_new_tokens présent
    assert "max_length" not in m.generation_kwargs
    assert m.generation_kwargs["max_new_tokens"] == 8


def test_unsupported_task_raises_value_error():
    with pytest.raises(ValueError):
        _ = Model(name="google/flan-t5-small", task="text2text-generation")


def test_output_col_collision_is_suffixed(patch_transformers):
    df = pd.DataFrame(
        {"index": [0], "full_note": ["x"], "full_note__facebook_opt_350m": ["exists"]}
    )
    ds = FakeDataset(df)

    m = Model(name="facebook/opt-350m", task="text-generation")
    out_ds = m.run(ds, prompt="p")
    # collision => suffix _1
    cols = [
        c for c in out_ds.data.columns if c.startswith("full_note__facebook_opt_350m")
    ]
    assert "full_note__facebook_opt_350m" in cols
    assert "full_note__facebook_opt_350m_1" in cols


def test_run_respects_explicit_output_col(patch_transformers):
    df = pd.DataFrame({"index": [0, 1], "text": ["x", "y"]})
    ds = FakeDataset(df, field="text")

    m = Model(name="repo/model", task="text-generation")
    out_ds = m.run(ds, "p", output_col="text_out")
    assert "text_out" in out_ds.data.columns
    assert list(out_ds.data["text_out"]) == ["GEN_OUT", "GEN_OUT"]


def test_errors_dataset_without_dataframe(patch_transformers):
    class BadDs:
        def __init__(self):
            self.data = {"index": [0], "full_note": ["x"]}
            self.field = "full_note"

    m = Model(name="x/y", task="text-generation")
    with pytest.raises(TypeError):
        m.run(BadDs(), "p")


def test_errors_missing_field_attribute(patch_transformers):
    class BadDs:
        def __init__(self, df):
            self.data = df

    df = pd.DataFrame({"index": [0], "full_note": ["x"]})
    m = Model(name="x/y", task="text-generation")
    with pytest.raises(ValueError):
        m.run(BadDs(df), "p")


def test_errors_missing_column(patch_transformers):
    df = pd.DataFrame({"index": [0], "other": ["x"]})

    class Ds:
        def __init__(self, df):
            self.data = df
            self.field = "full_note"

    ds = Ds(df)

    m = Model(name="x/y", task="text-generation")
    with pytest.raises(KeyError):
        m.run(ds, "p")


def test_pipeline_return_full_text_not_overridden(patch_transformers):
    # Si l'utilisateur a déjà mis pipeline_return_full_text=True,
    # run() ne doit PAS injecter return_full_text=False dans infer_kwargs
    df = pd.DataFrame({"index": [0, 1], "full_note": ["u", "v"]})
    ds = FakeDataset(df)

    m = Model(
        name="facebook/opt-350m",
        task="text-generation",
        pipeline_return_full_text=True,  # impose le comportement pipeline
    )

    _ = m.run(ds, "p")

    # On vérifie le dernier appel pipeline enregistré par le recorder
    call = m._pipe.calls[-1]
    # On ne veut PAS voir 'return_full_text' dans les kwargs d'inférence
    assert "return_full_text" not in call["infer_kwargs"]


def test_run_handles_dict_response(monkeypatch, patch_transformers):
    """Couvre la branche dict: result = {'generated_text': 'GEN_OUT'}."""
    from cleanote import model as model_mod

    class DictPipelineRecorder:
        def __init__(self, task, model, tokenizer, **kwargs):
            self.calls = []

        def __call__(self, inputs, **infer_kwargs):
            self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
            return {"generated_text": "GEN_OUT"}  # <-- dict, pas list

    monkeypatch.setattr(
        model_mod,
        "pipeline",
        lambda task, model, tokenizer, **kw: DictPipelineRecorder(
            task, model, tokenizer, **kw
        ),
    )

    df = pd.DataFrame({"full_note": ["a", "b"]})
    ds = FakeDataset(df)
    m = Model(name="repo/model", task="text-generation")

    out_ds = m.run(ds, "p")
    new_col = [c for c in out_ds.data.columns if c.startswith("full_note__")][0]
    assert list(out_ds.data[new_col]) == ["GEN_OUT", "GEN_OUT"]


def test_run_handles_non_list_non_dict_response(monkeypatch, patch_transformers):
    """Couvre la branche fallback: result est une str (ni list, ni dict)."""
    from cleanote import model as model_mod

    class StrPipelineRecorder:
        def __init__(self, task, model, tokenizer, **kwargs):
            self.calls = []

        def __call__(self, inputs, **infer_kwargs):
            self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
            return "PLAIN"  # <-- ni list ni dict

    monkeypatch.setattr(
        model_mod,
        "pipeline",
        lambda task, model, tokenizer, **kw: StrPipelineRecorder(
            task, model, tokenizer, **kw
        ),
    )

    df = pd.DataFrame({"full_note": ["a"]})
    ds = FakeDataset(df)
    m = Model(name="repo/model", task="text-generation")

    out_ds = m.run(ds, "p")
    new_col = [c for c in out_ds.data.columns if c.startswith("full_note__")][0]
    assert list(out_ds.data[new_col]) == ["PLAIN"]


def test_run_generation_overrides(monkeypatch, patch_transformers):
    """Vérifie que les overrides à l'appel (ex: max_new_tokens) sont bien pris en compte."""
    from cleanote import model as model_mod

    class RecordingPipeline:
        def __init__(self, task, model, tokenizer, **kwargs):
            self.calls = []

        def __call__(self, inputs, **infer_kwargs):
            self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
            return [{"generated_text": "OVERRIDE_OK"}]

    rec = RecordingPipeline
    monkeypatch.setattr(
        model_mod,
        "pipeline",
        lambda task, model, tokenizer, **kw: rec(task, model, tokenizer, **kw),
    )

    df = pd.DataFrame({"full_note": ["x"]})
    ds = FakeDataset(df)
    m = Model(name="repo/model", task="text-generation", max_new_tokens=128)

    out_ds = m.run(ds, "p", max_new_tokens=3)  # <-- override
    new_col = [c for c in out_ds.data.columns if c.startswith("full_note__")][0]
    assert list(out_ds.data[new_col]) == ["OVERRIDE_OK"]

    # On récupère le dernier appel pipeline et on vérifie l'override
    last_call = m._pipe.calls[-1]
    assert last_call["infer_kwargs"]["max_new_tokens"] == 3


def test_dtype_normalization_for_pipeline_and_model(monkeypatch):
    # On patch minimalement ce qu'il faut
    from cleanote import model as model_mod

    monkeypatch.setattr(
        model_mod,
        "AutoTokenizer",
        type(
            "X",
            (),
            {"from_pretrained": staticmethod(lambda name, **kw: FakeTokenizer())},
        ),
    )
    monkeypatch.setattr(
        model_mod,
        "AutoModelForCausalLM",
        type(
            "Y",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda name, **kw: FakeCausalModel(name, **kw)
                )
            },
        ),
    )

    # pipeline recorder simple
    class _P:
        def __init__(self, *a, **kw):
            self.calls = []

        def __call__(self, *a, **kw):
            self.calls.append((a, kw))
            return [{"generated_text": "ok"}]

    monkeypatch.setattr(model_mod, "pipeline", lambda *a, **kw: _P())

    # On passe dtype (pipeline) et model_dtype (model) pour activer _normalize_dtypes
    m = Model(
        name="x/y",
        task="text-generation",
        dtype="float16",  # -> doit devenir torch_dtype dans pipeline_kwargs
        model_dtype="float16",  # -> doit devenir torch_dtype dans model_kwargs
    )

    # Vérifie la normalisation
    assert "dtype" not in m.pipeline_kwargs
    assert m.pipeline_kwargs.get("torch_dtype") == "float16"
    assert "dtype" not in m.model_kwargs
    assert m.model_kwargs.get("torch_dtype") == "float16"


def test_ensure_pad_token_sets_from_eos(monkeypatch):
    from cleanote import model as model_mod

    # On garde une référence directe au tokenizer retourné
    tok = FakeTokenizer(pad_token_id=None, eos_token_id=42)  # pas de pad au départ

    monkeypatch.setattr(
        model_mod,
        "AutoTokenizer",
        type("X", (), {"from_pretrained": staticmethod(lambda name, **kw: tok)}),
    )
    monkeypatch.setattr(
        model_mod,
        "AutoModelForCausalLM",
        type(
            "Y",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda name, **kw: FakeCausalModel(name, **kw)
                )
            },
        ),
    )

    class _P:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            return [{"generated_text": "ok"}]

    monkeypatch.setattr(model_mod, "pipeline", lambda *a, **kw: _P())

    _ = Model(name="repo/model", task="text-generation")

    # Comme FakeTokenizer.eos_token = "<eos>", _ensure_pad_token doit l'avoir recopié
    assert tok.pad_token == "<eos>"


def test_generation_overrides_take_precedence(monkeypatch):
    from cleanote import model as model_mod

    class RecP:
        def __init__(self, *a, **kw):
            self.calls = []

        def __call__(self, inputs, **infer_kwargs):
            self.calls.append(infer_kwargs)
            return [{"generated_text": "ok"}]

    monkeypatch.setattr(
        model_mod,
        "AutoTokenizer",
        type(
            "X",
            (),
            {"from_pretrained": staticmethod(lambda name, **kw: FakeTokenizer())},
        ),
    )
    monkeypatch.setattr(
        model_mod,
        "AutoModelForCausalLM",
        type(
            "Y",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda name, **kw: FakeCausalModel(name, **kw)
                )
            },
        ),
    )
    rec = RecP()
    monkeypatch.setattr(model_mod, "pipeline", lambda *a, **kw: rec)

    df = pd.DataFrame({"full_note": ["a"]})
    ds = FakeDataset(df)

    m = Model(
        name="repo/model", task="text-generation", max_new_tokens=128, temperature=0.5
    )
    _ = m.run(ds, "p", max_new_tokens=3, temperature=0.0)

    # L'override doit primer
    assert rec.calls[-1]["max_new_tokens"] == 3
    assert rec.calls[-1]["temperature"] == 0.0


def test_load_is_idempotent(patch_transformers):
    m = Model(name="repo/model", task="text-generation")
    first_pipe = m._pipe
    # On appelle load() à nouveau -> doit retourner directement, sans recréer un nouveau pipe
    m.load()
    assert m._pipe is first_pipe


def test_run_raises_if_no_data_attr(patch_transformers):
    class BadDs:
        field = "x"  # pas d'attribut .data

    m = Model(name="repo/model", task="text-generation")
    with pytest.raises(ValueError) as e:
        m.run(BadDs(), "prompt")
    assert "No attribute 'data'" in str(e.value)
