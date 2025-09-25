class Pipeline:
    def __init__(self, dataset, model_h, model_v):
        self.dataset = dataset
        self.model_h = model_h
        self.model_v = model_v

    def apply(self):
        print("[Pipeline] Starting pipeline...")

        prompt_h = "please give me the number of words in the following text:"
        print("[Pipeline] Start Homogenization...")

        out_h_col = f"{self.dataset.field}__h"
        dataset_h = self.model_h.run(self.dataset, prompt_h, output_col=out_h_col)
        print("[Pipeline] Homogenization completed.")

        # prompt_v = "please verify the following text is not empty:"
        # out_v_col = f"{self.dataset.field}__v"
        # dataset_v = self.model_v.run(dataset_h, prompt_v, output_col=out_v_col)
        # print("[Pipeline] Verification completed.")
        # return dataset_v

        print("[Pipeline] Pipeline completed.")
        return dataset_h
