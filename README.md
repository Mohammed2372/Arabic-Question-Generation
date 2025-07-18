# Arabic Question Generation Project

## Overview

This project focuses on automatic question generation in Arabic using deep learning and NLP techniques. It leverages datasets such as Arabic-SQuAD, ARCD, MLQA, and TydiQA to train and evaluate models for generating high-quality, answerable questions from Arabic context passages.

## Project Structure

- **Notebooks**:
  - `arabic-question-generation.ipynb`: Main notebook for question generation experiments and evaluation.
  - `arabic-question-generation-preprocessing.ipynb`: Data loading, cleaning, and preprocessing steps.
  - `arabic-question-generation-train and predict.ipynb`: Model training and prediction pipeline.
  - `try with another data/`: Additional experiments with ARCD and TydiQA datasets.
- **Scripts**:
  - `test script.py`: Utility functions for text preprocessing and testing.
- **Results**:
  - `Generated Questions.pdf`, `Model test result with bert score and answerability.pdf`: Output and evaluation reports.

## Key Features

- Preprocessing and normalization of Arabic text (diacritics removal, punctuation, spacing, Alef variations).
- Utilizes transformer models (T5, mT5) for question generation.
- Evaluation using BLEU, ROUGE, and BERT-based metrics.
- Supports multiple Arabic QA datasets.

## How to Run

1. **Install Requirements**: Install all dependencies using the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```
   (You can still see the first cells in the notebooks for additional details.)
2. **Data Preparation**: Place the datasets in the `data/` directory as structured above.
3. **Run Notebooks**: Follow the order: preprocessing → training/prediction → main experiments.
4. **Testing**: Use `test script.py` for standalone text preprocessing or model testing.

## Example Usage

See the notebooks for step-by-step code and explanations. Example context and generated questions are provided in the results files.

## References

- [Arabic-SQuAD](https://www.kaggle.com/datasets/mohammed237/arabic-squad-processed)
- [ARCD](https://www.kaggle.com/datasets/mohammed237/arcd-dataset)
- [MLQA](https://www.kaggle.com/datasets/mohammed237/mlqa-data)
- [TydiQA](https://www.kaggle.com/datasets/mohammed237/tydiqa-data)
