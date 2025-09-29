import re
import json
import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
import pandas as pd


# ---------- Utils robustes de parsing JSON d'entités ----------
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
OBJ_NON_GREEDY_RE = re.compile(r"\{.*?\}", re.DOTALL)
EXPECTED_KEYS = ("Symptoms", "Diagnoses", "Treatments")


def _is_valid_schema(obj: Dict[str, Any]) -> bool:
    return (
        isinstance(obj, dict)
        and all(k in obj for k in EXPECTED_KEYS)
        and all(isinstance(obj[k], list) for k in EXPECTED_KEYS)
    )


def parse_json_entities(text: str) -> Optional[Dict[str, List[str]]]:
    """Tente plusieurs stratégies pour extraire un JSON valide {Symptoms[], Diagnoses[], Treatments[]}."""
    if not text:
        return None

    # 1) Essai direct
    try:
        obj = json.loads(text)
        if _is_valid_schema(obj):
            return obj
    except Exception:
        pass

    # 2) Bloc fenced ```json ... ```
    m = FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            if _is_valid_schema(obj):
                return obj
        except Exception:
            pass

    # 3) Tous les objets {...} non-gourmands
    for m in OBJ_NON_GREEDY_RE.finditer(text):
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if _is_valid_schema(obj):
                return obj
        except Exception:
            continue

    return None


# ---------- Pipeline ----------
class Pipeline:
    """
    Hypothèses légères :
      - self.dataset.data est un pandas.DataFrame
      - self.dataset.field est le nom de la colonne texte à traiter
      - model_h.run(dataset, prompt, output_col=...) renvoie un objet avec attribut .data (DataFrame) contenant la colonne de sortie
    """

    def __init__(self, dataset, model_h, model_v=None, verbose: bool = True):
        self.dataset = dataset
        self.model_h = model_h
        # self.model_v = model_v
        self.verbose = verbose

        # seront remplis au fil des étapes
        self.dataset_h = None  # dataset après substep1
        self._col_text = getattr(self.dataset, "field", "text")  # garde
        self._col_entities_raw = f"{self._col_text}__h_raw"
        self._col_summary = f"{self._col_text}__summary"

    # ----------------- API publique -----------------
    def apply(self) -> pd.DataFrame:
        """Exécute la pipeline et retourne un DataFrame final avec textes, entités (listes) et résumé."""
        if self.verbose:
            print("[Pipeline] Starting pipeline...")

        df_final = self.homogenize()

        if self.verbose:
            print("[Pipeline] Pipeline completed.")
        return df_final

    # ----------------- Étape 1 : extraction entités + Étape 2 : résumé -----------------
    def homogenize(self) -> pd.DataFrame:
        if self.verbose:
            print("[Pipeline - Homogenize] Start SS1 (entities extraction)...")
        df_with_entities = self.substep1()
        if self.verbose:
            print("[Pipeline - Homogenize] SS1 done.")

        if self.verbose:
            print("[Pipeline - Homogenize] Start SS2 (summarization)...")
        df_final = self.substep2(df_with_entities)
        if self.verbose:
            print("[Pipeline - Homogenize] SS2 done.")
        return df_final

    # ----------------- Substep 1 : extraction entités pour TOUTES les lignes -----------------
    def substep1(self) -> pd.DataFrame:
        prompt_h_ss1 = self.build_substep1_prompt()

        # Gardes dataset
        if not hasattr(self.dataset, "data"):
            raise ValueError("dataset.data manquant")
        if self.dataset.data.empty:
            # Colonnes vides prêtes pour la suite
            base = self.dataset.data.copy()
            base["Symptoms"] = [[] for _ in range(len(base))]
            base["Diagnoses"] = [[] for _ in range(len(base))]
            base["Treatments"] = [[] for _ in range(len(base))]
            return base

        # Nettoyage mémoire (ordre recommandé : gc puis CUDA)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.verbose:
            print("[Pipeline - SubStep1] Running model for entity extraction...")
        # Exécution modèle : on suppose que .run gère l'itération sur toutes les lignes
        self.dataset_h = self.model_h.run(
            self.dataset, prompt_h_ss1, output_col=self._col_entities_raw
        )

        # Sécurité minimale
        if not hasattr(self.dataset_h, "data"):
            raise RuntimeError(
                "model_h.run(...) n'a pas renvoyé un objet avec .data (DataFrame)."
            )
        df = self.dataset_h.data
        if self._col_entities_raw not in df.columns:
            raise RuntimeError(
                f"Colonne de sortie {self._col_entities_raw} absente du DataFrame renvoyé."
            )

        # Dump de la 1ère sortie brute pour debug
        try:
            print(
                "[DEBUG] raw_h_output_example:\n",
                self.dataset_h.data[self._col_entities_raw].iloc[0],
            )
        except Exception as e:
            print("[DEBUG] impossible d'afficher raw output:", e)

        # Parse pour chaque ligne
        symptoms_list: List[List[str]] = []
        diagnoses_list: List[List[str]] = []
        treatments_list: List[List[str]] = []

        for raw in df[self._col_entities_raw].astype(str).tolist():
            parsed = parse_json_entities(raw)
            if parsed is None:
                parsed = {"Symptoms": [], "Diagnoses": [], "Treatments": []}
            # cast propre en listes de str
            symptoms_list.append([str(x) for x in parsed.get("Symptoms", [])])
            diagnoses_list.append([str(x) for x in parsed.get("Diagnoses", [])])
            treatments_list.append([str(x) for x in parsed.get("Treatments", [])])

        # Ajoute colonnes entités
        df_entities = df.copy()
        df_entities["Symptoms"] = symptoms_list
        df_entities["Diagnoses"] = diagnoses_list
        df_entities["Treatments"] = treatments_list

        return df_entities

    # ----------------- Substep 2 : résumé pour TOUTES les lignes -----------------
    def substep2(self, df_entities: pd.DataFrame) -> pd.DataFrame:
        """
        Pour injecter les entités spécifiques à chaque ligne dans le prompt,
        on compose une nouvelle colonne texte : ENTITIES + NOTE.
        Puis on met temporairement dataset_h.field sur cette colonne.
        """
        # Compose une vue ligne-à-ligne "entities + note"
        ent_json_series = df_entities.apply(
            lambda r: json.dumps(
                {
                    "Symptoms": r.get("Symptoms", []),
                    "Diagnoses": r.get("Diagnoses", []),
                    "Treatments": r.get("Treatments", []),
                },
                ensure_ascii=False,
            ),
            axis=1,
        )

        composed_col = f"{self._col_text}__summary_input"
        df_entities = df_entities.copy()
        df_entities[composed_col] = (
            ent_json_series.radd("Extracted Entities (JSON):\n").radd("")
            + "\n\nClinical Note:\n"
            + df_entities[self._col_text].astype(str)
        )

        # On fabrique un petit "dataset" compatible avec model_h.run
        # Hypothèse : le type de self.dataset est un simple wrapper (data, field)
        # On en crée une copie modifiée.
        dataset_for_summary = (
            type(self.dataset)(
                dataset=(
                    df_entities
                    if "dataset" in type(self.dataset).__init__.__code__.co_varnames
                    else None
                ),  # si constructeur custom
                model_h=None,
                model_v=None,
            )
            if False
            else self.dataset
        )  # fallback : on réutilise self.dataset

        # Plus simple/robuste : on met à jour directement les attributs attendus.
        # (On évite d'appeler le constructeur si on ne le connaît pas.)
        dataset_for_summary = self.dataset_h  # réutilise l'objet renvoyé par substep1
        dataset_for_summary.data = df_entities
        dataset_for_summary.field = (
            composed_col  # le modèle lira ce champ comme 'entrée texte'
        )

        prompt_h_ss2 = (
            self.build_substep2_prompt()
        )  # prompt générique, ENTITIES déjà dans l'entrée

        # Mémoire
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.verbose:
            print("[Pipeline - SubStep2] Running model for summarization...")
        dataset_after_summary = self.model_h.run(
            dataset_for_summary, prompt_h_ss2, output_col=self._col_summary
        )

        df_out = dataset_after_summary.data
        if self._col_summary not in df_out.columns:
            raise RuntimeError(
                f"Colonne de sortie {self._col_summary} absente après summarization."
            )

        # DataFrame final minimal et lisible
        final_df = pd.DataFrame(
            {
                "text": df_out[self._col_text].astype(str),
                "Symptoms": df_out["Symptoms"],
                "Diagnoses": df_out["Diagnoses"],
                "Treatments": df_out["Treatments"],
                "Summary": df_out[self._col_summary].astype(str),
            }
        )

        return final_df

    # ----------------- Prompts -----------------
    def build_substep1_prompt(self) -> str:
        # La NOTE est lue via dataset.field par le modèle.
        return (
            "System: You are OpenBioLLM, a clinical text analysis assistant.\n"
            "Return ONLY a JSON object (no prose) with EXACTLY these keys:\n"
            '  "Symptoms": list of strings\n'
            '  "Diagnoses": list of strings\n'
            '  "Treatments": list of strings\n'
            "If none, use empty lists. Do not include explanations.\n"
            "Wrap the JSON in a ```json fenced code block.\n\n"
            "User: Extract the clinical entities from the clinical note below."
        )

    def build_substep2_prompt(self) -> str:
        # Le JSON des entités et la note sont déjà concaténés dans la colonne composed_col.
        return (
            "System: You are OpenBioLLM, a clinical summarization assistant.\n"
            "You will receive a JSON of extracted entities followed by the clinical note.\n"
            "Write a concise, clinically faithful summary weaving key symptoms, diagnoses, and treatments.\n"
            "Output ONLY the summary text (no JSON, no commentary)."
        )
