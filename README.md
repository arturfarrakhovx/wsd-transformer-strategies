# Comparative Analysis of Transformer-based Architectures and Strategies for Word Sense Disambiguation

## Overview

This repository contains the implementation and scientific report for a semester project in the **Computational Intelligence Methods** course.

The project addresses the challenge of **Word Sense Disambiguation (WSD)** – the task of automatically identifying the correct meaning of a word in a specific context. The goal was to move beyond simple baselines by implementing and comparing various Encoder-based Transformer architectures, ranging from standard fine-tuning to more complex strategies involving search space constraints and definition incorporation.

A complete pipeline was developed, starting from constructing a structured dataset from the **SemCor** corpus to implementing custom training loops. The final Gloss-augmented DeBERTa model achieved an F1 score of **74.3%**, significantly outperforming standard approaches and frequency-based heuristics.

## Approach

### Data Preprocessing
The training dataset was constructed from the **SemCor** corpus using the `nltk` library. A custom processing script was implemented to transform the corpus from its native chunked format into a structured dataset suitable for token classification.

-   **Sentence Reconstruction:** The pipeline dynamically reconstructs the full raw text while simultaneously tracking character-level indices to map target words to their corresponding sub-word tokens after BERT-style tokenization.
-   **Label Mapping:** Lemmas were mapped to specific WordNet synset names to create a consistent label space.
-   **Format:** The final processed data contains sentences, target words, character indices for target words, and synset labels with unique integer identifiers.
-   **Evaluation Data:** Standard Unified Evaluation Framework benchmarks (Senseval-2, Senseval-3, SemEval-2007, SemEval-2013, SemEval-2015) were sourced from an external repository in JSONL format. A preprocessing routine was developed to align these with the training data, involving label normalization from Sense Keys to Synset names and recalculation of character offsets.

### Architectures & Strategies
Three primary Transformer models were fine-tuned: **BERT**, **RoBERTa**, and **DeBERTa** (all base versions).  
Three distinct strategies were investigated:

1.  **Standard Token Classification:** The models were fine-tuned as token classifiers where the problem was treated as a multi-class classification task over the entire vocabulary (25k+ synsets).
2.  **Logit Masking:** A modification to the DeBERTa model where the output logits were masked before softmax. A mask vector was constructed for each target word, assigning valid candidate senses (derived from WordNet) a value of 0 and all others a large negative value. This constrained the search space strictly to linguistically plausible meanings.
3.  **Gloss-Augmented Architecture:** A modification of the DeBERTa model, in which the task was reformulated as a sentence-pair classification problem. The input sequence was constructed by concatenating the context sentence (with the target word marked by `[TGT]` tokens) and a candidate definition (gloss). A binary classifier was then trained to predict the relevance score of the gloss. A weighted Cross-Entropy Loss was implemented to handle the severe class imbalance between correct and incorrect candidates.

## Results

The models were evaluated against the standard **Unified Evaluation Framework** benchmarks using the Micro-averaged F1 Score (which in this case is equivalent to the Accuracy score).

-   **Baseline Performance (F1 on ALL dataset):**
    -   Lesk Algorithm (Knowledge-based): 33.6%
    -   Most Frequent Sense (MFS): 49.1%
    -   MFS+POS (Part of Speech) Filtering: 58.4%

-   **Transformer Performance (F1 on ALL dataset):**
    -   Standard BERT: 59.2%
    -   Standard RoBERTa: 58.9%
    -   Standard DeBERTa: 58.8%
    -   DeBERTa + Logit Masking: 65.0%
    -   **Gloss-Augmented DeBERTa: 74.3%**

-   **Key Findings:** Standard fine-tuning yielded results comparable to the MFS+POS heuristic due to the difficulty of discriminating across a vast output space. Logit Masking improved performance by +6% by constraining the search space. The Gloss-augmented approach achieved the highest performance (+15% over standard fine-tuning), validating the efficacy of explicitly modeling context-gloss relationships, which makes it possible to approach the performance of more complex SOTA systems such as ESCHER (80.7%) and ConSeC (82.0%).

## Usage

To run the code, first install the dependencies using `pip install -r requirements.txt`.

The project is structured into several stages:
1.  **Data Preparation:** Run `wsd_dataset_creation.ipynb` first. This script downloads the required `nltk` data (SemCor, WordNet, etc.) and generates the training dataset and the label map.
2.  **Data Analysis:** Run `data_analysis.ipynb` to view dataset statistics, POS distributions, and class imbalance.
3.  **Model Training:** The models are located in the `models/` directory. The training process varies slightly depending on the architecture.
    - **General Step**: Before training any model, copy `semcor_train.parquet` and `label_map.json` from the root directory into the specific model's subfolder.
    - **BERT & RoBERTa:** These models were trained using Google Colab. The code in `bert/` and `roberta/` folders is contained in Jupyter Notebooks and may require adjustments to match your local environment.
    - **DeBERTa (Standard & Masked):** These models were trained on a ClusterFIT environment. Their folders contain Python training scripts (`train_*.py`), execution scripts (`run_*.sh`) for the Slurm workload manager, and output logs (`.out`). You will need to edit the scripts to match your system configuration.
    - **Gloss DeBERTa:** First, run `gloss_dataset_creation.ipynb` to generate the gloss-augmented dataset specific to this architecture. Then, proceed with the training scripts provided in the folder (similar to the other DeBERTa models).
4.  **Evaluation:**
    - Convert evaluation data using `evaluation_data/evaluation_data_converting.ipynb` (converting initial JSONL into a format that models work with).
    - To prepare evaluation data for the Gloss model, run `evaluation_data/converting_to_gloss.ipynb`.
    - Run `evaluation.ipynb` to obtain final performance metrics and `unmasked_vs_masked.ipynb` for comparison of masked and unmasked strategies.

## Technology Stack
- **Programming Language:** Python
- **Environment:** Jupyter Notebook, ClusterFIT (Remote university computing cluster using `Slurm`).
- **Data Processing:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Natural Language Processing:**
  - **NLTK:** SemCor corpus, access to WordNet lexical database, text processing tools.
  - **Hugging Face Transformers:** Pre-trained models (BERT, RoBERTa, DeBERTa), Tokenizers, Trainer API.
- **Machine Learning & Deep Learning:**
  - **PyTorch:** Adaptation of model architectures and loss functions for WSD tasks, tensor operations.
  - **Scikit-learn:** Data splitting while keeping records related to the same target in one set for the gloss-augmented approach (`GroupShuffleSplit`).

## Repository Structure

The repository is organized as follows:  
```
report.pdf                                # Report in the format of a scientific article
dev/                                      # Directory with implementation code and data
├── wsd_dataset_creation.ipynb            # Dataset generation and NLTK setup  
├── semcor_train.parquet                  # Pre-processed training dataset  
├── label_map.json                        # Mapping of sense labels to IDs  
├── data_analysis.ipynb                   # Exploratory data analysis of the WSD dataset  
├── models/                               # Source code for different architectures  
│   ├── bert/                             # BERT implementation (Jupyter Notebooks for Colab)  
│   ├── roberta/                          # RoBERTa implementation (Jupyter Notebooks for Colab)  
│   ├── deberta/                          # DeBERTa implementation (Python & Slurm scripts)  
│   ├── deberta_masked/                   # Masked DeBERTa (Python & Slurm scripts)  
│   └── deberta_gloss/                    # Gloss-augmented DeBERTa (Dataset creator & scripts)  
├── evaluation_data/                      # Benchmark datasets  
│   ├── jsonl/                            # Raw evaluation data in JSONL format  
│   ├── evaluation_data_converting.ipynb  # Prepares standard evaluation data  
│   └── converting_to_gloss.ipynb         # Prepares gloss-augmented evaluation data  
├── evaluation.ipynb                      # Main evaluation notebook  
├── unmasked_vs_masked.ipynb              # Comparison of masked and unmasked strategies
└── results/                              # CSV files containing benchmark metrics and results  
```
