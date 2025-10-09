import logging
from typing import Iterable, Callable, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import pandas as pd
import numpy as np
import warnings

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
        logger.info("[Pipeline] Starting pipeline...")
        self.homogenize()
        self.verify_QuickUMLS()
        self.verify_NLI()
        logger.info("[Pipeline] Pipeline completed.")
        return self.dataset_h

    # ------------------------- Homogénéisation -------------------------

    def homogenize(self):
        logger.info("[Pipeline] Prompt for Homogenization:")
        if self.model_h.prompt:
            logger.info(self.model_h.prompt)
        else:
            self.model_h.prompt = self.build_prompt_h()
            logger.info(self.model_h.prompt)

        logger.info("[Pipeline] Start Homogenization...")
        out_h_col = f"{self.dataset.field}__h"
        self.dataset_h = self.model_h.run(self.dataset, output_col=out_h_col)
        logger.info("[Pipeline] Homogenization completed.")

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
        logger.info("[Pipeline] Starting QuickUMLS verification...")
        logger.info("[Pipeline] QuickUMLS verification completed.")

    def verify_NLI(self):
        logger.info("[Pipeline] Starting NLI verification...")
        self._ensure_spacy()
        self._ensure_nli()
        logger.info({"id2label": self._id2label})
        logger.info("[Pipeline] NLI verification completed.")

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
        return [sent.text.strip() for sent in doc.sents]

    def _ensure_spacy(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download

                logger.info("Downloading spaCy model: en_core_web_sm")
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
        # Optionnel: assurer la présence des trois clés
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

        logger.info(
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
