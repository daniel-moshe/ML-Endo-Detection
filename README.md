# Endometriosis Detection Using Machine Learning

[![Logo](https://github.com/daniel-moshe/ML-Endo-Detection/blob/main/Docs/logo.png?raw=true "Logo")](https://github.com/daniel-moshe/ML-Endo-Detection/blob/main/Docs/logo.png?raw=true "Logo")

## Overview

This project focuses on the early detection of endometriosis using machine learning techniques. By analyzing patient data from the UK Biobank, we aim to create a predictive model that assists healthcare professionals in diagnosing endometriosis earlier, potentially reducing diagnostic delays and improving patient outcomes.

## Project Structure

The project is organized as follows:

### 1. `Code/`

- **`Dataset/`**: Contains the scripts and data files used for processing the input dataset.
  - `features_data.csv`: A table containing feature names and their corresponding UKB code.
  - `features_data.csv.pkl`: A pickled version for faster loading.
  - `parse_database.py`: Script for parsing the raw data on the UKB server into a merged dataset.
  
- **`Model/`**: Includes the scripts for model training and evaluation.
  - `best_estimator.py`: The script for identifying the best performing model.
  - `create_cohort.py`: Used for cohort selection and grouping the dataset.
  - `features_preprocess.py`: Handles feature preprocessing such as normalization or scaling.
  - `model_selection.py`: Implements model selection and cross-validation logic.
  - `utils.py`: Contains utility functions used throughout the model scripts.
  
- **`Notebooks/`**: Jupyter notebooks used for exploration and experimentation.
  - `best_model.ipynb`: Notebook that showcases the best model found.
  - `playground_cohort.ipynb`: Notebook for experimenting with different cohort selections and data manipulations.

### 2. `Docs/`

- `Features/`: Contains documentation on feature ideas and categorizations of features used in the models.
- `Poster/`: Contains the project's poster and its visual assets, such as logos, diagrams, and model visualizations.
- `Visualizations/`: Graphical representations of various features used in the model.
- **Proposal and Summary Documents**: These documents summarize the goals, mid-term docs, and final results of the project.
- **Project Presentations**: PDF and PowerPoint presentations explaining the project's progress and results.

## Usage

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/daniel-moshe/ML-Endo-Detection.git
    cd ML-Endo-Detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Results
- **Best Model**:  CatBoost was identified as the best-performing model.
- **Model Accuracy**: Achieved an accuracy of 73% and an F1 score of 72% on the test set.
- **SHAP Analysis**: The SHAP analysis revealed that features `other_gyno_conditions`, and `estrogen_exposure` had the most significant impact on the model's predictions.

## License

This project is licensed under the MIT License.
