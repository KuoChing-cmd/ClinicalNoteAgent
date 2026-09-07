# ClinicalNoteAgent

Welcome to the **ClinicalNoteAgent** repository! This codebase is designed for multimodal clinical decision support, integrating structured Electronic Health Record (EHR) data with unstructured clinical notes. Our models leverage advanced deep learning techniques, including ClinicalBERT, to predict outcomes such as patient readmission.

## 🚀 Features

- **Multimodal Learning**: Combines time-series medical data (vitals, labs), static demographics, and free-text clinical notes.
- **End-to-End ClinicalBERT Fine-Tuning**: Fine-tune domain-specific language models (like `emilyalsentzer/Bio_ClinicalBERT`) dynamically during downstream tasks.
- **Flexible Architectures**: Support for both Early Fusion and Late Fusion modeling strategies.
- **Retrieval-Augmented Generation (RAG)**: Built-in retrievers for evidence-based note extraction and processing.

## 📂 Repository Structure

- `scripts/preprocessing/` 
  - Scripts for processing clinical notes, extracting ClinicalBERT embeddings, and RAG retrievers.
- `scripts/training/`
  - Core training loop, PyTorch data loaders, configurations, metrics calculation, and unified model definitions.
- `scripts/analysis/`
  - Model evaluation tools and feature importance analysis.
- `tests/`
  - End-to-end smoke tests verifying the training pipelines.

## 🛠 Setup & Installation

1. Clone the repository and install dependencies:
   ```bash
   git clone <repository_url>
   cd ClinicalNoteAgent
   pip install -r requirements.txt
   ```

2. Prepare your EHR datasets (e.g., MIMIC-III, MIMIC-IV) and place them in the appropriate data directories.

## 📈 Training

To start training an end-to-end fusion model with clinical notes:
```bash
python scripts/training/train.py --config <your_config_file.yaml>
```
For more complex configurations, refer to the files in `scripts/training/config.py`.

## 📜 License & Citation

*(License information to be added upon official code release)*
