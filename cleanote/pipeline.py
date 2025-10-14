import logging
from typing import Iterable, Callable, Dict, List, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import pandas as pd
import numpy as np
import warnings
import json
import re


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

        self.SCI_MODEL_NAME = "en_core_sci_lg"  # ou "en_core_sci_sm"
        self.UMLS_MIN_SCORE = 0.0
        self._sci = None
        self._umls_cache = {}

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

    def _ensure_scispacy(self):
        if self._sci is not None:
            return self._sci

        try:
            import scispacy  # noqa: F401
            from scispacy.umls_linking import UmlsEntityLinker
        except Exception as e:
            raise RuntimeError(
                "SciSpaCy n'est pas installé (pip install scispacy)."
            ) from e

        # charge le modèle (lg -> fallback sm)
        try:
            self._sci = spacy.load(self.SCI_MODEL_NAME)
        except OSError:
            try:
                self._sci = spacy.load("en_core_sci_sm")
                print("[UMLS] Fallback → en_core_sci_sm (lg non installé)")
            except OSError as e:
                raise RuntimeError(
                    "Aucun modèle SciSpaCy trouvé. Installe en_core_sci_lg ou en_core_sci_sm."
                ) from e

        # ajoute le linker UMLS
        if "scispacy_linker" in self._sci.pipe_names:
            self._sci.remove_pipe("scispacy_linker")
        self._sci.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "k": 30,
                "threshold": float(self.UMLS_MIN_SCORE),
            },
            last=True,
        )
        return self._sci

    def _norm_term(self, t: str) -> str:
        return re.sub(r"\s+", " ", (t or "").strip().lower())

    def _umls_match_bulk(self, terms: List[str]) -> Dict[str, bool]:
        """Retourne {terme_original -> True/False} selon présence d'au moins un CUI (score >= seuil)."""
        self._ensure_scispacy()
        # normalise + uniques
        mapping = {t: self._norm_term(t) for t in terms}
        uniq_norm = sorted(set(v for v in mapping.values() if v))
        to_run = [u for u in uniq_norm if u not in self._umls_cache]

        if to_run:
            docs = self._sci.pipe(to_run, batch_size=64, n_process=1)
            for key, doc in zip(to_run, docs):
                ok = False
                for ent in doc.ents:
                    kb = getattr(ent._, "kb_ents", [])
                    if kb:
                        best = max(score for _, score in kb)
                        if best >= float(self.UMLS_MIN_SCORE):
                            ok = True
                            break
                self._umls_cache[key] = ok

        # re-projette sur les termes d'origine
        return {t: self._umls_cache.get(mapping[t], False) for t in terms}

    def _extract_entity_list(self, row, field_name: str, out_h_col: str) -> List[str]:
        """
        Récupère la liste d'entités pour field_name ('Symptoms'/'MedicalConclusion'/'Treatments').
        Essaie d'abord la colonne directe; sinon parse le JSON dans out_h_col.
        """
        vals = row.get(field_name)
        # déjà une liste ?
        if isinstance(vals, list):
            return [str(x).strip() for x in vals if str(x).strip()]
        # string: tente JSON, sinon split simple
        if isinstance(vals, str) and vals.strip():
            try:
                parsed = json.loads(vals)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                parts = re.split(r"[;,]\s*", vals.strip())
                return [p for p in parts if p]
        # fallback: JSON homogénéisé
        payload = row.get(out_h_col)
        try:
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict):
                arr = payload.get(field_name, [])
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
        return []

    def verify_QuickUMLS(self):
        """
        Vérification via SciSpaCy + UMLS linker.
        Pour chaque colonne (Symptoms, MedicalConclusion, Treatments), ajoute:
        *_umls_total, *_umls_matched, *_umls_match_rate, *_umls_loss_rate
        """
        print("[Pipeline] Starting UMLS verification (SciSpaCy linker)...")
        self._ensure_scispacy()

        df = self.dataset_h.data
        out_h_col = f"{self.dataset.field}__h"
        triplets = [
            ("Symptoms", "symptoms"),
            ("MedicalConclusion", "medicalconclusion"),
            ("Treatments", "treatments"),
        ]

        # crée colonnes résultat si absentes
        for _, short in triplets:
            for suffix in (
                "umls_total",
                "umls_matched",
                "umls_match_rate",
                "umls_loss_rate",
            ):
                col = f"{short}_{suffix}"
                if col not in df.columns:
                    df[col] = np.nan

        for idx, row in df.iterrows():
            for field, short in triplets:
                entities = self._extract_entity_list(row, field, out_h_col)
                total = len(entities)

                if total == 0:
                    matched = 0
                    match_rate = np.nan
                    loss_rate = np.nan
                else:
                    status = self._umls_match_bulk(entities)  # {term: bool}
                    matched = sum(1 for t in entities if status.get(t, False))
                    match_rate = matched / total
                    loss_rate = 1.0 - match_rate

                df.at[idx, f"{short}_umls_total"] = total
                df.at[idx, f"{short}_umls_matched"] = matched
                df.at[idx, f"{short}_umls_match_rate"] = match_rate
                df.at[idx, f"{short}_umls_loss_rate"] = loss_rate

            if (idx + 1) % 10 == 0:
                print(f"[UMLS] processed {idx+1}/{len(df)} rows...")

        print("[Pipeline] UMLS verification completed.")

    def verify_NLI(self):
        print("[Pipeline] Starting NLI verification...")
        self._ensure_nli()

        df = self.dataset_h.data
        text_col = self.dataset.field
        out_h_col = f"{self.dataset.field}__h"

        # Colonnes résultats
        for col in ("nli_ent_mean", "nli_neu_mean", "nli_con_mean"):
            if col not in df.columns:
                df[col] = np.nan

        for idx, row in df.iterrows():
            # Texte source (comme src_sents dans Colab)
            src_text = (row.get(text_col) or "").strip()

            # Résumé/hypothèses (comme sum_sents dans Colab)
            summ_text = (row.get("Summary") or "").strip()
            if not summ_text and out_h_col in df.columns:
                try:
                    payload = row[out_h_col]
                    payload = (
                        json.loads(payload)
                        if isinstance(payload, str)
                        else (payload or {})
                    )
                    summ_text = (payload or {}).get("Summary", "") or ""
                except Exception:
                    pass
            summ_text = summ_text.strip()

            if not src_text or not summ_text:
                print(f"[Pipeline] Row {idx}: texte ou résumé vide, skip.")
                continue

            premises = self.decouper_texte_en_phrases(src_text)  # source
            hypotheses = self.decouper_texte_en_phrases(summ_text)  # résumé

            if not premises or not hypotheses:
                print(f"[Pipeline] Row {idx}: pas de phrases, skip.")
                continue

            # Comme Colab: lignes = résumé, colonnes = source; nli(premise=src, hypothesis=sum)
            matrice = self.generer_table(
                hypotheses,
                premises,
                lambda h, p: self.nli(p, h, return_probs=True),
            )

            avg = self.average(hypotheses, premises, matrice)

            df.at[idx, "nli_ent_mean"] = avg["entailment"]
            df.at[idx, "nli_neu_mean"] = avg["neutral"]
            df.at[idx, "nli_con_mean"] = avg["contradiction"]

            print(
                f"[Pipeline] Row {idx} → entail={avg['entailment']}, neutral={avg['neutral']}, contra={avg['contradiction']}"
            )

        print("[Pipeline] NLI verification completed.")

    # _ensure_nlp : sans newline_boundaries
    def _ensure_nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                self._nlp = spacy.blank("en")
            if "sentencizer" in self._nlp.pipe_names:
                self._nlp.remove_pipe("sentencizer")
            self._nlp.add_pipe(
                "sentencizer",
                config={"punct_chars": [".", "!", "?"]},
                first=True,
            )

    def nli(self, premise: str, hypothesis: str, return_probs: bool = True) -> Dict:
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

        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "prediction": pred_label,
            "probs": (
                dict(zip(labels, [round(float(p), 4) for p in probs]))
                if return_probs
                else None
            ),
        }

    # ------------------------- spaCy utils -------------------------

    def decouper_texte_en_phrases(self, texte: str) -> List[str]:
        self._ensure_nlp()
        txt = self._normalize_for_sentences(texte or "")
        doc = self._nlp(txt)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    # normalisation des sauts de ligne (remplace ta version)
    def _normalize_for_sentences(self, texte: str) -> str:
        if not texte:
            return ""
        t = texte.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
        # ponctuation + \n -> ponctuation + espace
        t = re.sub(r"([.!?])\s*\n\s*", r"\1 ", t)
        # \n sans ponctuation avant -> point + espace
        t = re.sub(r"(?<![.!?])\s*\n\s*", ". ", t)
        # espaces multiples -> 1
        t = re.sub(r"\s{2,}", " ", t).strip()
        return t

    # ------------------------- NLI model load -------------------------

    def _ensure_nli(self):
        if self._tok is None or self._clf is None:
            self._tok = AutoTokenizer.from_pretrained(self.NLI_MODEL_NAME)
            self._clf = AutoModelForSequenceClassification.from_pretrained(
                self.NLI_MODEL_NAME
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clf = self._clf.to(self.device)
            self._id2label = {
                i: lbl.lower() for i, lbl in self._clf.config.id2label.items()
            }
        return self._tok, self._clf, self._id2label

    # ------------------------- Tableaux & métriques -------------------------

    def generer_table(
        self,
        lignes: Iterable,
        colonnes: Iterable,
        fonction: Callable[[Any, Any], Dict],
    ) -> List[List[Dict]]:
        return [[self._prettier(fonction(i, j)) for j in colonnes] for i in lignes]

    @staticmethod
    def _prettier(res: Dict) -> Dict:
        probs = (res or {}).get("probs", {}) or {}
        # Les clés sont déjà normalisées en minuscules dans nli()
        for k in ("entailment", "neutral", "contradiction"):
            probs.setdefault(k, None)
        return probs

    def average(self, lignes: Iterable, colonnes: Iterable, matrice: List[List[Dict]]):
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
            f"Moyennes — entailment={mean_best_ent}, neutral={mean_best_neu}, contradiction={mean_best_con}"
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
