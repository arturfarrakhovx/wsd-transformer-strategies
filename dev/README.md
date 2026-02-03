# Comparative Analysis of Transformer-based Architectures and Strategies for Word Sense Disambiguation

## Project Assignment
**Word Sense Disambiguation.** Prepare a suitable dataset, fine-tune and compare different architectures, and benchmark the best-performing solution against existing WSD systems.

## Dependencies Installation

To run the code in this repository, you need to install the required dependencies. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Usage Instructions

Follow the steps below to reproduce the training and evaluation process.

### 1. Data Preparation

First, run the `wsd_dataset_creation.ipynb` notebook.

- **Note**: Although the generated dataset (`semcor_train.parquet`) and the label map (`label_map.json`) are already included in the root of the repository, this notebook must be executed to download the necessary NLTK data files required for subsequent steps.

### 2. Data Analysis

Run `data_analysis.ipynb` to inspect the dataset statistics and properties.

### 3. Model Training

The models are located in the `models/` directory. The training process varies slightly depending on the architecture.

**General Step**: Before training any model, copy `semcor_train.parquet` and `label_map.json` from the root directory into the specific model's subfolder.

**Model-Specific Instructions**:

- **BERT & RoBERTa**: These models were trained using Google Colab. The code in `bert/` and `roberta/` folders is contained in Jupyter Notebooks and may require adjustments to match your local environment.

- **DeBERTa (Standard & Masked)**: These models were trained on a ClusterFIT environment. Their folders contain Python training scripts (`train_*.py`), execution scripts (`run_*.sh`) for the Slurm workload manager, and output logs (`.out`). You will need to edit the scripts to match your system configuration.

- **Gloss DeBERTa**: First, run `gloss_dataset_creation.ipynb` to generate the gloss-augmented dataset specific to this architecture. Then, proceed with the training scripts provided in the folder (similar to the other DeBERTa models).

### 4. Evaluation Preparation

Before running the benchmarks, the evaluation data must be converted into the correct format.

1. Run `evaluation_data/evaluation_data_converting.ipynb`.
2. Run `evaluation_data/converting_to_gloss.ipynb` (required for the Gloss architecture).

### 5. Final Evaluation

Finally, execute the following notebooks to obtain performance metrics:

- `evaluation.ipynb`: Main evaluation script.
- `unmasked_vs_masked.ipynb`: Comparison between masked and unmasked strategies.

## Repository Structure

The repository is organized as follows:  
```
.
├── data_analysis.ipynb                   # Exploratory data analysis of the WSD dataset  
├── evaluation.ipynb                      # Main evaluation notebook  
├── evaluation_data/                      # Benchmark datasets  
│   ├── jsonl/                            # Raw data in JSONL format  
│   ├── evaluation_data_converting.ipynb  # Prepares standard evaluation data  
│   └── converting_to_gloss.ipynb         # Prepares gloss-augmented evaluation data  
├── label_map.json                        # Mapping of sense labels to IDs  
├── models/                               # Source code for different architectures  
│   ├── bert/                             # BERT implementation (Jupyter Notebooks for Colab)  
│   ├── roberta/                          # RoBERTa implementation (Jupyter Notebooks for Colab)  
│   ├── deberta/                          # DeBERTa implementation (Python & Slurm scripts)  
│   ├── deberta_masked/                   # Masked DeBERTa strategy (Python & Slurm scripts)  
│   └── deberta_gloss/                    # Gloss-augmented DeBERTa (Dataset creator & scripts)  
├── report.pdf                            # Final project report  
├── results/                              # CSV files containing benchmark metrics and results  
├── semcor_train.parquet                  # Pre-processed training dataset  
├── unmasked_vs_masked.ipynb              # Comparison of masked and unmasked strategies
└── wsd_dataset_creation.ipynb            # Dataset generation and NLTK setup  
```