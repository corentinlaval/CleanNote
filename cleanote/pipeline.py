import re
import json
import gc
import torch


class Pipeline:
    def __init__(self, dataset, model_h, model_v):
        self.dataset = dataset
        self.model_h = model_h
        self.model_v = model_v
        self.dataset_h = None
        self.dataset_v = None

    def apply(self):
        print("[Pipeline] Starting pipeline...")

        out_h = self.homogenize()

        # prompt_v = "please verify the following text is not empty:"
        # out_v_col = f"{self.dataset.field}__v"
        # dataset_v = self.model_v.run(dataset_h, prompt_v, output_col=out_v_col)
        # print("[Pipeline] Verification completed.")
        # return dataset_v

        print("[Pipeline] Pipeline completed.")
        return out_h

    def homogenize(self) -> str:
        print("[Pipeline - Homogenize] Start Homogenization SS1...")
        out_h_ss1 = self.substep1()
        print("[Pipeline - Homogenize] Homogenization SS1 Done.")

        print("[Pipeline - Homogenize] Start Homogenization SS2...")
        out_h_ss2 = self.substep2(out_h_ss1)
        print("[Pipeline - Homogenize] Homogenization SS2 Done.")
        return out_h_ss2

    def substep1(self) -> dict:
        prompt_h_ss1 = self.build_substep1_prompt()
        max_retries = 3

        for attempt in range(1, max_retries + 1):

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            print(f"[Pipeline - SubStep1] Start Homogenization {attempt}...")

            out_h_col = f"{self.dataset.field}__h"

            self.dataset_h = self.model_h.run(
                self.dataset, prompt_h_ss1, output_col=out_h_col
            )

            out_h_ss1 = self.dataset_h.data.loc[0, out_h_col]

            match = re.search(r"\{.*\}", out_h_ss1, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if (
                        isinstance(parsed.get("Symptoms"), list)
                        and isinstance(parsed.get("Diagnoses"), list)
                        and isinstance(parsed.get("Treatments"), list)
                    ):
                        return parsed
                except json.JSONDecodeError:
                    pass
            print(f"Retry {attempt}/{max_retries} for JSON extraction...")

            print(f"[Pipeline - SubStep1] Homogenization {attempt} completed.")
        return {"Symptoms": [], "Diagnoses": [], "Treatments": []}

    def substep2(self, entities: dict) -> str:
        prompt_h_ss2 = self.build_substep2_prompt(entities)
        out_h_ss2_col = f"{self.dataset.field}__v"
        self.dataset_h = self.model_h.run(
            self.dataset_h, prompt_h_ss2, output_col=out_h_ss2_col
        )
        out_h_ss2 = self.dataset_h.data.loc[0, out_h_ss2_col]
        return out_h_ss2

    def build_substep1_prompt(self) -> str:
        return (
            "System: You are OpenBioLLM, a clinical text analysis assistant.\n"
            "Output ONLY a JSON object with exactly these keys:\n"
            '- "Symptoms": list of strings (empty if none)\n'
            '- "Diagnoses": list of strings (empty if none)\n'
            '- "Treatments": list of strings (empty if none)\n'
            "Do not add any other keys or text.\n\n"
            "User: Extract the clinical entities from the note below:\n"
        )

    def build_substep2_prompt(self, entities: dict) -> str:
        ent_json = json.dumps(
            {
                "Symptoms": entities["Symptoms"],
                "Diagnoses": entities["Diagnoses"],
                "Treatments": entities["Treatments"],
            },
            ensure_ascii=False,
        )
        return (
            "System: You are OpenBioLLM, a clinical summarization assistant.\n"
            "You will be given a clinical note and a JSON of extracted entities.\n"
            "Write a concise summary of the note, weaving in the key symptoms, diagnoses, and treatments.\n"
            "Output ONLY the summary text, no other JSON or commentary.\n\n"
            f"Extracted Entities:\n{ent_json}\n\n"
            "Clinical Note:"
        )
