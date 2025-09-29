class Pipeline:
    def __init__(self, dataset, model_h):
        self.dataset = dataset
        self.model_h = model_h
        self.dataset_h = None

    def apply(self):
        print("[Pipeline] Starting pipeline...")

        self.homogenize()

        print("[Pipeline] Pipeline completed.")

        return self.dataset_h

    def homogenize(self):

        prompt_h = self.build_prompt_h()
        print("[Pipeline] Prompt for Homogenization:")
        print(prompt_h)

        print("[Pipeline] Start Homogenization...")

        out_h_col = f"{self.dataset.field}__h"

        self.dataset_h = self.model_h.run(self.dataset, prompt_h, output_col=out_h_col)

        print("[Pipeline] Homogenization completed.")
        return

    def build_prompt_h(self) -> str:
        return """Analyze the document below and return a single, valid JSON object with exactly these keys (no trailing commas):

            {{
            "Symptoms": [/* list of symptoms extracted from the document */],
            "MedicalConclusion": [/* list of medical conclusion extracted from the document */],
            "Treatments": [/* list of treatments extracted from the document */],
            "Summary": "/* a professional paragraph summarizing the note, that mentions only items already listed in Symptoms, MedicalConclusion, and Treatments, without inventing anything */"
            }}

            - If no information exists for a given key, return an empty array for that key.  
            - The Summary field **must use only** the items already extracted above and **must not add** any new facts.  
            - Ensure the output is **syntactically valid JSON**.

            Document:
            """

    def to_excel(self) -> str:
        self.dataset_h.data.to_excel("dataset_h.xlsx", index=False)
        return "dataset_h.xlsx"
