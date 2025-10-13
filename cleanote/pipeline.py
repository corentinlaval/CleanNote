import logging
from typing import Iterable, Callable, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import pandas as pd
import numpy as np
import warnings
import json

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Pipeline de prétraitement/validation.
    `dataset` doit exposer:
      - `field`: nom de la colonne texte à traiter
      - `data`: DataFrame (utilisée par to_excel)
    `model_h` doit exposer:
      - `prompt`: str | None
      - `run(dataset, output_col=...)` -> retourne un objet compatible avec `dataset_h`
    """

    NLI_MODEL_NAME = "pritamdeka/PubMedBERT-MNLI-MedNLI"

    def __init__(self, dataset, model_h):
        self.dataset = dataset
        self.model_h = model_h
        self.dataset_h = None

        # Cache / état
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._nlp = None  # spaCy nlp
        self._tok = None  # HF tokenizer
        self._clf = None  # HF sequence classification model
        self._id2label: Optional[Dict[int, str]] = None

    # ------------------------- Orchestration -------------------------

    def apply(self):
        print("[Pipeline] Starting pipeline...")
        self.homogenize()
        self.verify_QuickUMLS()
        self.verify_NLI()
        print("[Pipeline] Pipeline completed.")
        return self.dataset_h

    # ------------------------- Homogénéisation -------------------------

    def homogenize(self):
        print("[Pipeline] Prompt for Homogenization:")
        if self.model_h.prompt:
            print(self.model_h.prompt)
        else:
            self.model_h.prompt = self.build_prompt_h()
            print(self.model_h.prompt)

        print("[Pipeline] Start Homogenization...")
        out_h_col = f"{self.dataset.field}__h"
        self.dataset_h = self.model_h.run(self.dataset, output_col=out_h_col)
        print("[Pipeline] Homogenization completed.")

    @staticmethod
    def build_prompt_h() -> str:
        # Retourne un prompt JSON clair et valide
        return (
            "Analyze the document below and return a single, valid JSON object with exactly these keys:\n"
            "{\n"
            '  "Symptoms": [],\n'
            '  "MedicalConclusion": [],\n'
            '  "Treatments": [],\n'
            '  "Summary": ""\n'
            "}\n"
            "- If no information exists for a given key, return an empty array for that key.\n"
            "- The Summary must only use items already extracted above (no new facts).\n"
            "- Ensure the output is syntactically valid JSON.\n"
            "Document:\n"
        )

    # ------------------------- Vérifications -------------------------

    def verify_QuickUMLS(self):
        # Placeholder: branchement futur à QuickUMLS
        print("[Pipeline] Starting QuickUMLS verification...")
        print("[Pipeline] QuickUMLS verification completed.")

    def verify_NLI(self):
        print("[Pipeline] Starting NLI verification...")
        self._ensure_spacy()
        self._ensure_nli()
        print({"id2label": self._id2label})

        df = self.dataset_h.data
        print(f"[Pipeline] Processing {len(df)} rows...")
        print(df.head(2))
        text_col = self.dataset.field
        out_h_col = f"{self.dataset.field}__h"

        # Ajout des colonnes de sortie si absentes
        for col in ("nli_ent_mean", "nli_neu_mean", "nli_con_mean"):
            if col not in df.columns:
                df[col] = np.nan

        for idx, row in df.iterrows():
            text = str(row.get(text_col, "") or "").strip()
            if not text:
                print(f"[Pipeline] Row {idx} has empty text, skipping NLI.")
                continue

            # Découper le texte en phrases
            premises = self.decouper_texte_en_phrases(text)
            if not premises:
                print(f"[Pipeline] Row {idx} has no sentences, skipping NLI.")
                continue

            # Extraire les hypothèses à partir du JSON homogénéisé
            hypotheses = self._extract_hypotheses(row.get(out_h_col))
            print(row.get(out_h_col))
            if not hypotheses:
                print(f"[Pipeline] Row {idx} has no hypotheses, skipping NLI.")
                continue

            # Calculer la matrice NLI (chaque phrase vs chaque hypothèse)
            matrice = self.generer_table(
                premises,
                hypotheses,
                lambda p, h: self.nli(p, h, return_probs=True),
            )

            # Calculer la moyenne des meilleures hypothèses par phrase
            avg = self.average(premises, hypotheses, matrice)

            # Ajouter les moyennes dans le DataFrame
            df.at[idx, "nli_ent_mean"] = avg["entailment"]
            print("entailment: ", avg["entailment"])
            df.at[idx, "nli_neu_mean"] = avg["neutral"]
            print("neutral: ", avg["neutral"])
            df.at[idx, "nli_con_mean"] = avg["contradiction"]
            print("contradiction: ", avg["contradiction"])

        print("[Pipeline] NLI verification completed.")

    def _extract_hypotheses(self, h_json) -> List[str]:
        """Récupère Summary + listes depuis le JSON homogénéisé."""
        if not h_json:
            return []
        try:
            data = json.loads(h_json) if isinstance(h_json, str) else h_json
        except Exception:
            return []

        if not isinstance(data, dict):
            return []

        hypotheses = []
        summary = str(data.get("Summary") or "").strip()
        if summary:
            hypotheses.append(summary)

        for key in ("Symptoms", "MedicalConclusion", "Treatments"):
            val = data.get(key)
            if isinstance(val, list):
                hypotheses.extend([str(v).strip() for v in val if str(v).strip()])

        # Supprimer les doublons
        return list(dict.fromkeys(hypotheses))

    # ------------------------- NLI utils -------------------------

    def nli(self, premise: str, hypothesis: str, return_probs: bool = True) -> Dict:
        """Retourne la prédiction NLI et (optionnellement) les probabilités."""
        self._ensure_nli()

        inputs = self._tok(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        self._clf.eval()
        with torch.inference_mode():
            logits = self._clf(**inputs).logits.squeeze(0)

        probs_t = torch.softmax(logits, dim=-1).cpu()
        probs = probs_t.tolist()
        labels = [self._id2label[i] for i in range(len(probs))]
        pred_idx = int(torch.argmax(probs_t))
        pred_label = self._id2label[pred_idx]

        res = {
            "premise": premise,
            "hypothesis": hypothesis,
            "prediction": pred_label,
            "probs": (
                dict(zip(labels, [round(float(p), 4) for p in probs]))
                if return_probs
                else None
            ),
        }
        return res

    # ------------------------- spaCy utils -------------------------

    def decouper_texte_en_phrases(self, texte: str) -> List[str]:
        nlp = self._ensure_spacy()
        doc = nlp(texte)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _ensure_spacy(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download

                print("Downloading spaCy model: en_core_web_sm")
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    # ------------------------- NLI model load -------------------------

    def _ensure_nli(self):
        if self._tok is None or self._clf is None or self._id2label is None:
            self._tok = AutoTokenizer.from_pretrained(self.NLI_MODEL_NAME)
            self._clf = AutoModelForSequenceClassification.from_pretrained(
                self.NLI_MODEL_NAME
            ).to(self.device)
            self._id2label = self._clf.config.id2label
        return self._tok, self._clf, self._id2label

    # ------------------------- Tableaux & métriques -------------------------

    def generer_table(
        self,
        lignes: Iterable,
        colonnes: Iterable,
        fonction: Callable[[any, any], Dict],
    ) -> List[List[Dict]]:
        """Construit une matrice en appliquant fonction(ligne, colonne) et renvoie les dict résultats."""
        return [[self._prettier(fonction(i, j)) for j in colonnes] for i in lignes]

    @staticmethod
    def _prettier(res: Dict) -> Dict:
        """Nettoie/valide une cellule { 'probs': {...} } -> retourne le dict des probs."""
        probs = (res or {}).get("probs", {})
        for k in ("entailment", "neutral", "contradiction"):
            probs.setdefault(k, None)
        return probs

    def average(self, lignes: Iterable, colonnes: Iterable, matrice: List[List[Dict]]):
        """Calcule les moyennes des meilleures colonnes (entailment max par ligne)."""
        df = pd.DataFrame(matrice, index=list(lignes), columns=list(colonnes))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            entailments = df.applymap(
                lambda x: x.get("entailment") if isinstance(x, dict) else None
            )

        best_col_per_row = entailments.idxmax(axis=1)

        best_ent_vals, best_neu_vals, best_con_vals = [], [], []
        for i in df.index:
            best_col = best_col_per_row.loc[i]
            cell = df.loc[i, best_col]
            if isinstance(cell, dict):
                best_ent_vals.append(cell.get("entailment"))
                best_neu_vals.append(cell.get("neutral"))
                best_con_vals.append(cell.get("contradiction"))

        mean_best_ent = float(np.nanmean(best_ent_vals)) if best_ent_vals else None
        mean_best_neu = float(np.nanmean(best_neu_vals)) if best_neu_vals else None
        mean_best_con = float(np.nanmean(best_con_vals)) if best_con_vals else None

        print(
            "Moyennes — entailment=%s, neutral=%s, contradiction=%s",
            mean_best_ent,
            mean_best_neu,
            mean_best_con,
        )
        return {
            "entailment": mean_best_ent,
            "neutral": mean_best_neu,
            "contradiction": mean_best_con,
        }

    # ------------------------- Export -------------------------

    def to_excel(self) -> str:
        """Exporte le DataFrame `dataset_h.data` en Excel."""
        path = "dataset_h.xlsx"
        self.dataset_h.data.to_excel(path, index=False)
        return path
