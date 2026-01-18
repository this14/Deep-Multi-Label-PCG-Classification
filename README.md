# Deep-Multi-Label-PCG-Classification
# Deep Multi-Label Classification of Pediatric Heart Murmur Attributes

**Official Repository for the Manuscript:** *"Deep Multi-Label Classification of Pediatric Heart Murmur Attributes (Timing, Shape, Grading) Using Multi-Location Phonocardiogram Data"*

## 📌 Overview
This study presents a deep learning approach for the simultaneous multi-label classification of heart murmur attributes: **Timing** (Systolic/Diastolic phases), **Shape** (Crescendo/Decrescendo/etc.), and **Grading** (Intensity). We utilized the **CirCor DigiScope Dataset** to evaluate two distinct architectures:

1.  **Multi-Branch Fusion Model**: A complex architecture integrating audio signals from four auscultation locations (AV, PV, TV, MV).
2.  **Single-Location Baseline**: A streamlined model using only the Aortic Valve (AV) signal (Demonstrated superior generalizability on this dataset).

## 📂 Repository Structure
This repository allows full reproducibility of the results presented in the paper. The files are organized into three main directories corresponding to the Supplementary Information (S1-S3).

### 1. Source Code (`/SI_Code_S1`)
Contains all Python scripts required for data analysis, model training, and evaluation.
* `Analysis.py`: Initial exploration of the dataset statistics (Table 1 generation).
* `train_model_augmented.py`: Training script for the **Multi-Branch Fusion Model**.
* `train_model_single_location.py`: Training script for the **Single-Location Baseline Model** (Recommended).
* `compare_models.py`: Script to generate comparison metrics and confusion matrices.
* `requirements.txt`: List of dependencies.

### 2. Model Weights (`/SI_ModelWeights_S2`)
Pre-trained PyTorch weights (`.pth`) for the models discussed in the manuscript.
* `best_model.pth`: Weights for the Multi-Branch Fusion Model.
* `best_model_single.pth`: Weights for the Single-Location Baseline Model.

### 3. Experimental Results (`/SI_log_S3`)
Raw outputs, logs, and visualization figures.
* `test_results.csv`: Macro-F1 scores and accuracy metrics.
* `confusion_matrices.png`: Confusion matrices for Timing, Shape, and Grading tasks.
* `training_history.png`: Loss curves showing training vs. validation performance.
* `Compare of two models.docx` & `Supporting Information...`: Detailed training logs.

---

## 🚀 Getting Started

### Prerequisites
The code is implemented in **Python 3.8+** using **PyTorch**.
To set up the environment, install the required packages located in the S1 folder:

```bash
pip install -r SI_Code_S1/requirements.txt
