# 🩺 Skin Disease Classifier

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Live demo:** https://hm-ai-skin-classifier.streamlit.app/  
**Status:** Research MVP (not a medical device)

This app classifies common skin conditions from a photo and **abstains when uncertain** (returns `uncertain/healthy` below a confidence threshold).  
It’s designed for *cautious triage*, not diagnosis.

---

## Table of contents
- [Motivation](#motivation)
- [Demo & Features](#demo--features)
- [How it works](#how-it-works)
- [Data](#data)
- [Results](#results)
- [Install & Run](#install--run)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Evaluation & Tests](#evaluation--tests)
- [Safety & Limitations](#safety--limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Motivation
As a parent, I wanted a **cautious, accessible tool** to triage rashes and frequent conditions without overconfident guesses. This project grew from that need into a small, production-like MVP with a clean ML workflow, clear metrics, and a simple UI that anyone can try.

---

## Demo & Features
- 📷 **Upload or use sample images** (per-class gallery)  
- 🎚️ **Decision threshold slider** — trades coverage for precision  
- 🧠 **Uncertainty-aware**: returns `uncertain/healthy` when not confident  
- 📊 **Probabilities bar chart** + top-2 note when borderline  
- 🧾 **Conditions glossary** & **How to take a good photo** tips  
- ⚙️ **Reproducible training/evaluation** & basic tests

---
## How it works
---

---

## Data
- DermNet-based dataset. Examples of included classes: acne, contact_dermatitis, eczema, psoriasis, rash, scabies, tinea_ringworm, urticaria, warts.
- Custom stratified split 75/15/10 (train/val/test)
-  Please check the original data licensing/terms if you plan to use or redistribute the dataset. This project is educational/research-only

---

## Results
### Model Performance @ Confidence Threshold 0.75

| Split | Coverage | Confident Accuracy | Macro F1-Score | Total Samples |
|-------|----------|-------------------|----------------|---------------|
| **Validation** | 76.6% (2,294/2,994) | **97.38%** | **95.01%** | 2,994 |
| **Test** | 77.9% (2,315/2,970) | **97.54%** | **93.82%** | 2,970 |

### Threshold Analysis (Validation Set)
- Lower Threshold (0.50) → High Coverage (91.7%) but Lower Precision (94.1%)
- Higher Threshold (0.85) → Lower Coverage (64.5%) but Higher Precision (98.6%)

<details>
<summary><strong>📋 Complete Threshold Sweep Results</strong></summary>

| Threshold | Coverage | Confident Accuracy | Macro F1-Score |
|-----------|----------|-------------------|----------------|
| 0.50 | 91.7% | 94.10% | 90.63% |
| 0.55 | 89.7% | 94.83% | 91.66% |
| 0.60 | 86.8% | 95.50% | 92.38% |
| 0.65 | 84.3% | 96.23% | 93.46% |
| 0.70 | 81.3% | 96.92% | 94.75% |
| **0.75** | **76.6%** | **97.38%** | **95.01%** |
| 0.80 | 72.5% | 97.74% | 95.51% |
| 0.85 | 64.5% | 98.55% | 97.75% |
| 0.90 | 52.9% | 98.86% | 98.51% |

</details>

> **💡 Clinical Insight:** The default threshold (0.75) balances coverage and precision for practical triage scenarios. Higher thresholds prioritize accuracy over coverage, enabling "better safe than sorry" decision making in healthcare contexts.



---

## Install & Run

**Live demo:** https://hm-ai-skin-classifier.streamlit.app/  
**Requirements:** Python 3.11+, see `requirements.txt`

```bash
# 1) Clone & enter
git clone https://github.com/your-user/skin-disease-classifier-mvp.git
cd skin-disease-classifier-mvp

# 2) (Optional) create venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Run the app
streamlit run app.py
```
---
## Configuration
- This project is ENV-driven (see src/config.py).
- Create your own .env (not committed) or use .env.example as a template.

---
## Project Structure
```
├── app.py                      # Streamlit UI
├── artifacts/                  # best_resnet18.pt, classes.txt, metrics.csv
├── demo_samples/<class>/...    # small gallery for demo
├── src/
│   ├── __init__.py
│   ├── config.py               # ENV-driven config (.env supported)
│   ├── data.py                 # datasets/dataloaders + transforms
│   ├── model.py                # ResNet18 head swap
│   ├── train.py                # train loop (class weights + smoothing)
│   ├── evaluate.py             # coverage-aware evaluation + sweep
│   ├── infer.py                # cached model load + predict_pil
│   └── logging_utils.py        # stdout + rotating file handler
├── tests/                      # basic unit tests
├── requirements.txt
├── .env.example
└── .streamlit/config.toml      # theme
```
---
## Evaluation & Tests
Run unit tests:
```bash
python -m pytest -q

Included tests:
test_model_forward.py — verifies model forward pass shape.

test_transforms.py — checks preprocessing shape and normalized ranges.

test_evaluate_threshold.py — validates coverage/metrics logic used in evaluation.

test_threshold.py — ensures uncertainty behavior (uncertain/healthy) at high thresholds
(skips if artifacts are missing).
```
---
## Safety & Limitations
- Not a medical device — research/education only.
- Dataset bias: trained on public mixed-age images; not curated for clinical coverage.
- Image quality matters: poor lighting, blur, or distant crops degrade predictions.
- Abstention by design: the app returns uncertain/healthy when confidence < threshold.
- Out-of-distribution: rare conditions or atypical presentations may be misclassified or flagged as uncertain.
---
## Acknowledgements
- DermNet-sourced images via public Kaggle mirror (educational dermatology imagery). Please review original sources for licensing and usage constraints.
- Libraries: PyTorch, TorchVision, Streamlit, scikit-learn, and the open-source community.
---
## License
This project is released under the MIT License. See the LICENSE file for details.








