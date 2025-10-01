# CleanNote 
[![English](https://img.shields.io/badge/lang-English-blue)](#english)
[![Français](https://img.shields.io/badge/lang-Français-green)](#français)


![CI](https://github.com/corentinlaval/CleanNote/actions/workflows/ci.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/corentinlaval/CleanNote/branch/main/graph/badge.svg?branch=main)](https://codecov.io/gh/corentinlaval/CleanNote)
[![PyPI version](https://img.shields.io/pypi/v/cleanote.svg)](https://pypi.org/project/cleanote/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  


---
## English 
<p align="right">
  <a href="#cleannote">
    <img src="https://img.shields.io/badge/▲-Back_to_top-blue" alt="Back to top"/>
  </a>
</p>

**CleanNote** analyzes raw medical notes and transforms them into concise, structured documents focused on symptoms, medical conclusions, and treatments, enabling easier analysis and clinical research.

This solution was developed during a research internship at the **** laboratory, and is currently demonstrated using the **AGBonnet/Augmented-clinical-notes** dataset together with the **mistralai/Mistral-7B-Instruct-v0.3** model.

This work was supervised by three people:

- **Ms.** ****, company supervisor and associate professor, who proposed the topic of the preprocessing pipeline,

- **Ms.** ****, school supervisor and associate professor, who guided me on frugality aspects and formatting,

- **Mr.** ****, PhD student, who provided support on identifying adapted solutions and domain knowledge.

---

## Installation 
<p align="right">
  <a href="#cleannote">
    <img src="https://img.shields.io/badge/▲-Back_to_top-blue" alt="Back to top"/>
  </a>
</p>

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
<p align="right">
  <a href="#cleannote">
    <img src="https://img.shields.io/badge/▲-Back_to_top-blue" alt="Back to top"/>
  </a>
</p>

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

## Literature
<p align="right">
  <a href="#cleannote">
    <img src="https://img.shields.io/badge/▲-Back_to_top-blue" alt="Back to top"/>
  </a>
</p>

- *Identification de profils patients à partir de notes cliniques non structurées.*  
  **Corentin Laval, Catherine Combes, Rémi Eyraud, Virginie Fresse**.  
  *PFIA 2025.*


---

## Français 
<p align="right">
  <a href="#cleannote">
    <img src="https://img.shields.io/badge/▲-Haut_de_page-green" alt="Haut de page"/>
  </a>
</p>

**CleanNote** analyse des notes médicales brutes et les transforme en documents concis et structurés, centrés sur les symptômes, les conclusions médicales et les traitements, afin de faciliter leur analyse et la recherche clinique.

Cette solution a été développée dans le cadre d’un stage de recherche au laboratoire **** , et est actuellement démontrée à l’aide du jeu de données **AGBonnet/Augmented-clinical-notes** ainsi que du modèle **mistralai/Mistral-7B-Instruct-v0.3**.

Ce travail a été supervisé par trois personnes :

- **Mme** ****, tutrice entreprise et maîtresse de conférences, qui a proposé le sujet du pipeline de prétraitement,

- **Mme** ****, tutrice école et maîtresse de conférences, qui m’a encadré sur les aspects de frugalité et de mise en forme,

- **M.** ****, doctorant, qui m’a accompagné sur l’identification des solutions adaptées et l’apport de connaissances du domaine.

---

## Installation 
<p align="right">
  <a href="#cleannote">
    <img src="https://img.shields.io/badge/▲-Haut_de_page-green" alt="Haut de page"/>
  </a>
</p>

Tout d’abord, assurez-vous d’être dans votre environnement virtuel Python (par ex. `venv`).  
Pour installer la **dernière version disponible** (voir le badge PyPI ci-dessus) :

```bash
 pip install -U cleanote
```

Si vous souhaitez installer une version spécifique (par exemple `0.2.1`):

```bash
 pip install -U cleanote==0.2.1
```
La dernière version publiée est toujours affichée dans le badge PyPI en haut de ce README.

---

## Utilisation 
<p align="right">
  <a href="#cleannote">
    <img src="https://img.shields.io/badge/▲-Haut_de_page-green" alt="Haut de page"/>
  </a>
</p>

Après installation, vous pouvez utiliser **CleanNote** en seulement quelques lignes de code :

```bash
from cleanote.dataset import Dataset
from cleanote.model import Model
from cleanote.pipeline import Pipeline

# Charger un jeu de données
data = Dataset(name="AGBonnet/Augmented-clinical-notes", split="train", field="full_note", limit=1)

# Charger un modèle
model = Model(name="mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens=512)

# Créer le pipeline
pipe = Pipeline(dataset=data, model_h=model)

# Lancer le pipeline
out = pipe.apply()

# Afficher le résultat
print(out.data.head())

# Exporter le jeu de données homogénéisé
xls = pipe.to_excel()  
print(f"Fichier Excel sauvegardé : {xls}")
```

---

## Références
<p align="right">
  <a href="#cleannote">
    <img src="https://img.shields.io/badge/▲-Haut_de_page-green" alt="Haut de page"/>
  </a>
</p>

- *Identification de profils patients à partir de notes cliniques non structurées.*  
  **Corentin Laval, Catherine Combes, Rémi Eyraud, Virginie Fresse**.  
  *PFIA 2025.*


---

## License  
This project is licensed under the [MIT License](LICENSE).  
