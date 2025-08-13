# 🩺 Skin Condition Classifier

[![Open in Streamlit](https://img.shields.io/badge/Live%20demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://hm-ai-skin-classifier.streamlit.app/)
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
### Overall Flow
- **Image Input**: An image is either uploaded or selected from a sample gallery.
- **Preprocessing**: The image is resized, normalized, and its EXIF orientation is corrected.
- **Model Inference**: The processed image is fed into a **ResNet18** model.
- **Probability & Prediction**: The model outputs **Softmax probabilities**.
    - If the maximum probability is **less than a set threshold**, the result is classified as **"uncertain/healthy"**.
    - If the maximum probability is **greater than or equal to the threshold**, the corresponding class with the highest probability is the **predicted class**.
- **UI Output**: The user interface displays the result, along with a probability chart.

### Preprocessing
- **Training**: Images are augmented using `RandomResizedCrop`, `HorizontalFlip`, `Rotation`, and `ColorJitter`, followed by normalization with ImageNet mean/std.
- **Validation/Testing**: Images are simply resized and normalized.

### Model
- A **ResNet18** architecture is used, initialized with **ImageNet** pretrained weights.
- The final fully connected layer is replaced to match the number of classes.
- The model runs on the best available device (`auto/cuda/cpu`) with a deterministic seed.

### Training
- **Optimizer**: `AdamW` is used with a set learning rate (`LR`) and `weight_decay`.
- **Loss Function**: `CrossEntropyLoss` with label smoothing (0.05) and class weights (inverse frequency) is used.
- **Scheduler**: `ReduceLROnPlateau` is used to adjust the learning rate based on the validation macro-F1 score.
- **Artifacts**:
    - The best model is saved as `best_resnet18.pt`.
    - Class order is stored in `classes.txt`.
    - Per-epoch metrics (`epoch,val_acc,val_f1,lr`) are logged in `metrics.csv`.

### Evaluation
- **Process**: Logits are collected and converted to softmax probabilities.
- **Metrics**:
    - Metrics (`coverage`, `accuracy`, `macro-F1`) are reported on a "confident subset" of predictions (where `max_prob ≥ threshold`).
    - A threshold sweep from 0.50 to 0.90 is performed to analyze the trade-off between coverage and quality.
- **Reports**: A full `classification_report` and a confusion matrix are generated for all labels.

### Inference
- The model is loaded once and cached for efficiency.
- An input PIL image is preprocessed and passed to the model to get logits and probabilities.
- If the top probability is below the threshold, the output is "uncertain/healthy". Otherwise, it returns the predicted class.
- The UI also shows the top-2 probabilities and flags "borderline cases" where the difference between the top two probabilities is less than 0.05.

### UI (Streamlit)
- **Inputs**: A gallery of sample images for each class and a file uploader.
- **Controls**: A slider to adjust the prediction abstention threshold.
- **Outputs**:
    - A bar chart of prediction probabilities.
    - A glossary of conditions.
    - Tips for taking photos.

---

## Data
### Supported Conditions

The model classifies 9 common dermatological conditions with high accuracy:

| Condition | Clinical Description | Key Visual Features |
|-----------|---------------------|-------------------|
| **Acne** | Clogged hair follicles causing inflammatory lesions | Comedones, papules, pustules on face/back |
| **Contact Dermatitis** | Allergic or irritant reaction to external substances | Localized red, itchy rash at contact site |
| **Eczema** | Chronic inflammatory skin condition | Dry, scaly patches in flexural areas |
| **Psoriasis** | Autoimmune condition with rapid skin cell turnover | Well-demarcated plaques with silvery scales |
| **Rash** | General term for widespread skin inflammation | Non-specific redness, may be viral/bacterial |
| **Scabies** | Parasitic mite infestation causing intense itching | Linear burrows, especially web spaces/wrists |
| **Tinea/Ringworm** | Fungal infection of skin, hair, or nails | Ring-shaped lesions with raised, scaly borders |
| **Urticaria** | Allergic reaction causing temporary skin welts | Raised, itchy wheals that appear/disappear |
| **Warts** | Viral skin growths caused by HPV | Small, rough-textured papules |
- DermNet-based dataset. 
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
git clone (https://github.com/HMurawski/Skin_Condition_Classifier)
cd skin_condition_classifier

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











