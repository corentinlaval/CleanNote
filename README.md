# CleanNote

![CI](https://github.com/corentinlaval/CleanNote/actions/workflows/ci.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/corentinlaval/CleanNote/branch/main/graph/badge.svg?branch=main)](https://codecov.io/gh/corentinlaval/CleanNote)
[![PyPI version](https://img.shields.io/pypi/v/cleanote.svg)](https://pypi.org/project/cleanote/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  


**CleanNote** analyzes raw medical notes and transforms them into concise, structured documents focused on symptoms, medical conclusions, and treatments, enabling easier analysis and clinical research.

This solution was developed during a research internship at the **** laboratory, and is currently demonstrated using the **AGBonnet/Augmented-clinical-notes** dataset together with the **mistralai/Mistral-7B-Instruct-v0.3** model.

This work was supervised by three people:

- **Ms.** ****, company supervisor and associate professor, who proposed the topic of the preprocessing pipeline,

- **Ms.** ****, school supervisor and associate professor, who guided me on frugality aspects and formatting,

- **Mr.** ****, company supervisor and PhD student, who provided support on identifying adapted solutions and domain knowledge.

---

## Installation
First, make sure you are inside your Python virtual environment (e.g. `venv`).  
To install the **latest available version** (see the PyPI badge above):

```bash
 pip install -U cleanote
```

If you want to install a specific version (for example `0.2.1`):

```bash
 pip install -U cleanote==0.2.1
```
The latest released version is always displayed in the PyPI badge at the top of this README.

---

## Usage
After installation, you can using **CleanNote** with just few lines of code:

```bash
from cleanote.dataset import Dataset
from cleanote.model import Model
from cleanote.pipeline import Pipeline

# Load a dataset
data = Dataset(name="AGBonnet/Augmented-clinical-notes", split="train", field="full_note", limit=1)

# Load a model
model = Model(name="mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens=512)

# Create pipeline
pipe = Pipeline(dataset=data, model_h=model)

# Run pipeline
out = pipe.apply()

# Display result
print(out.data.head())

# Download the dataset homogenized
xls = pipe.to_excel()  
print(f"Excel file saved to : {xls}")

```

---

## Quickstart  

Clone the repository:  

```
git clone https://github.com/corentinlaval/CleanNote.git
cd CleanNote
python -m cleanote --config configs/base.yaml
```



---

## License  
This project is licensed under the [MIT License](LICENSE).  
