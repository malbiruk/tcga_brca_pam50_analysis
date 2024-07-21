# TCGA BRCA Dataset Analysis Based on PAM50 Subtypes

This repository contains analysis scripts and notebooks for the TCGA BRCA dataset based on PAM50 subtypes.

## Results

**Presentation** with the results is [here](https://insilico-test-task-slides.pages.dev/).

Controls:
- **F** to go fullscreen
- **Esc** to exit
- **Arrows**, **Space**, or **Mouse wheel** to navigate
- **G** then `1/{n of slide}` to go to specific slide

## Data

**HTSeq-FPKM-UQ** data from [here](https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443) was used.

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Directory Structure
- `results/`
  - `feature_importance/`: Contains feature importance results.
  - `diff_expression/`: Contains differential expression results.
  - `figures/`: Contains generated figures.
  - `classifiers_evaluation/`: Contains classifier evaluation results.
- `data/`
  - `processed/`: Contains processed data.
  - `input/`: Contains raw input data.
- `scripts/`
  - `__init__.py`: Initialization script for the scripts module.
  - `limma.py`: Script for running the limma analysis.

## Notebooks and Scripts
- `00_preprocessing.ipynb` / `00_preprocessing.py`:
  - **Purpose**: Preprocesses the TCGA-BRCA dataset.
  - **Steps**:
    - Loads and merges data tables.
    - Converts Ensembl IDs to gene names.
    - Filters the dataset based on mean gene expression values (across all samples).
    - Saves the filtered dataset and metadata.
- `01_differential_expression.ipynb` / `01_differential_expression.py`:
  - **Purpose**: Performs differential expression (DE) analysis.
  - **Steps**:
    - Creates experiment input files for each PAM50 subtype.
    - Runs the limma analysis for each experiment using `scripts/limma.py`.
    - Visualizes and describes DEGs using volcano plots.
- `02_dimension_reduction.ipynb` / `02_dimension_reduction.py`:
  - **Purpose**: Reduces the dimensions of the dataset and visualizes it.
  - **Steps**:
    - Applies PCA, t-SNE, and UMAP for dimension reduction.
    - Visualizes the results using scatter plots.
    - Creates subsets of the dataset based on top variable genes and DEGs.
- `03_classifier.ipynb` / `03_classifier.py`:
  - **Purpose**: Implements and evaluates classifiers for breast cancer subtypes.
  - **Steps**:
    - Trains and evaluates multiple classifiers (e.g., GradientBoosting, SVM, Random Forest).
    - Tunes hyperparameters for top classifiers.
    - Evaluates feature importance using various methods (e.g., impurity, permutation, SHAP).
