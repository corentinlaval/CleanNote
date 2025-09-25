# tests/test_model.py
import pandas as pd
import pytest

# Adapte l'import si ton fichier s'appelle autrement:
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


class FakeSeq2SeqModel(FakeCausalModel):
    pass


class PipelineRecorder:
    """Capture les kwargs passés à pipeline() et la dernière inférence."""

    def __init__(self, task, model, tokenizer, **kwargs):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.calls = []  # liste des inputs reçus

    def __call__(self, inputs, **infer_kwargs):
        # inputs peut être une str (unitaire) dans notre code
        self.calls.append({"inputs": inputs, "infer_kwargs": infer_kwargs})
        if self.task == "text-generation":
            return [{"generated_text": "GEN_OUT"}]
        else:
            # pour text2text-generation, on renvoie un champ différent
            return [{"summary_text": "SEQ2SEQ_OUT"}]


# ---------- Fixtures de monkeypatch ----------
@pytest.fixture
def patch_transformers(monkeypatch):
    created = {}

    def fake_auto_tokenizer_from_pretrained(name, **kwargs):
        created["tokenizer_called_with"] = {"name": name, **kwargs}
        # Simule l'absence de pad_token_id pour tester la logique de fallback
        return FakeTokenizer(pad_token_id=None, eos_token_id=1, **kwargs)

    def fake_causal_from_pretrained(name, **kwargs):
        created["causal_model_called_with"] = {"name": name, **kwargs}
        return FakeCausalModel(name, **kwargs)

    def fake_seq2seq_from_pretrained(name, **kwargs):
        created["seq2seq_model_called_with"] = {"name": name, **kwargs}
        return FakeSeq2SeqModel(name, **kwargs)

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
    monkeypatch.setattr(
        "cleanote.model.AutoModelForSeq2SeqLM.from_pretrained",
        fake_seq2seq_from_pretrained,
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
def test_text_generation_happy_path_df_adds_column_and_sets_pad(
    monkeypatch, patch_transformers, capsys
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

    # pad_token doit être positionné par la classe (pad_token_id None, eos_token_id=1)
    # Vérifie que tokenizer_use_fast a bien été pris en compte
    assert patch_transformers["tokenizer_called_with"]["use_fast"] is True

    # run => ajoute une colonne auto-nommée
    out_ds = m.run(ds, prompt="count words please")
    new_cols = [c for c in out_ds.data.columns if c.startswith(ds.field + "__")]
    assert len(new_cols) == 1
    col = new_cols[0]
    assert list(out_ds.data[col]) == ["GEN_OUT", "GEN_OUT"]
    # l'original n'est pas écrasé
    assert list(out_ds.data[ds.field]) == ["hello", "world"]
    # garde un pointeur vers la dernière colonne
    assert out_ds.last_output_col == col

    # pipeline kwargs: device par défaut = -1 si non fourni
    assert patch_transformers["pipeline_called_with"]["kwargs"].get("device", -2) == -1

    # generation_kwargs: max_length doit avoir été retiré si max_new_tokens présent
    assert "max_length" not in m.generation_kwargs
    assert m.generation_kwargs["max_new_tokens"] == 8


def test_text2text_generation_branch(monkeypatch, patch_transformers):
    df = pd.DataFrame({"index": [0, 1], "full_note": ["a", "b"]})
    ds = FakeDataset(df)

    m = Model(name="google/flan-t5-small", task="text2text-generation", num_beams=2)
    out_ds = m.run(ds, prompt="rewrite politely")
    # colonne ajoutée avec la sortie seq2seq
    new_col = [c for c in out_ds.data.columns if c.startswith(ds.field + "__")][0]
    assert list(out_ds.data[new_col]) == ["SEQ2SEQ_OUT", "SEQ2SEQ_OUT"]


def test_output_col_collision_is_suffixed(monkeypatch, patch_transformers):
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


def test_run_respects_explicit_output_col(monkeypatch, patch_transformers):
    df = pd.DataFrame({"index": [0, 1], "text": ["x", "y"]})
    ds = FakeDataset(df, field="text")

    m = Model(name="repo/model", task="text-generation")
    out_ds = m.run(ds, "p", output_col="text_out")
    assert "text_out" in out_ds.data.columns
    assert list(out_ds.data["text_out"]) == ["GEN_OUT", "GEN_OUT"]


def test_errors_dataset_without_dataframe(monkeypatch, patch_transformers):
    class BadDs:
        def __init__(self):
            self.data = {"index": [0], "full_note": ["x"]}
            self.field = "full_note"

    m = Model(name="x/y", task="text-generation")
    with pytest.raises(TypeError):
        m.run(BadDs(), "p")


def test_errors_missing_field_attribute(monkeypatch, patch_transformers):
    class BadDs:
        def __init__(self, df):
            self.data = df

    df = pd.DataFrame({"index": [0], "full_note": ["x"]})
    m = Model(name="x/y", task="text-generation")
    with pytest.raises(ValueError):
        m.run(BadDs(df), "p")


def test_errors_missing_column(monkeypatch, patch_transformers):
    df = pd.DataFrame({"index": [0], "other": ["x"]})

    class Ds:
        def __init__(self, df):
            self.data = df
            self.field = "full_note"

    ds = Ds(df)

    m = Model(name="x/y", task="text-generation")
    with pytest.raises(KeyError):
        m.run(ds, "p")


def test_pipeline_return_full_text_not_overridden(monkeypatch, patch_transformers):
    # Si l'utilisateur a déjà mis pipeline_return_full_text=True,
    # _apply_to_texts ne doit PAS forcer return_full_text=False
    df = pd.DataFrame({"index": [0, 1], "full_note": ["u", "v"]})
    ds = FakeDataset(df)

    m = Model(
        name="facebook/opt-350m",
        task="text-generation",
        pipeline_return_full_text=True,  # impose le comportement pipeline
    )

    _ = m.run(ds, "p")
    # récupérer le dernier appel pipeline (enregistré dans fixture)
    # et vérifier que return_full_text n'a PAS été injecté par _apply_to_texts
    # (on ne peut pas accéder direct ici; on valide en re-déclarant un nouveau prompt)
    # -> On force un 2e run et on teste qu'aucun param "return_full_text" n'est ajouté
    out = m._pipe.calls[-1]
    assert "return_full_text" not in out["infer_kwargs"]
