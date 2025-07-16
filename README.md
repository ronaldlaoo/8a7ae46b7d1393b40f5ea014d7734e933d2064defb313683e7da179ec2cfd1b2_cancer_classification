# 8a7ae46b7d1393b40f5ea014d7734e933d2064defb313683e7da179ec2cfd1b2_cancer_classification

## Project Overview  
The main objective of this repository is to develop a production driven DS project rather than focusing on improving the model accuracy. I selected the Breast Cancer dataset from sklearn because it's small in size, readily available in sklearn.dataset, and I personally want to explore more on health applications.

## How to Get the Data  
It is based on the [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) from sklearn. It is already provided in data/raw/cancer_dataset.csv for convenience, so no additional download is needed.

## Setup Instructions  
To run this project:

1. Create a virtual environment using [UV](https://github.com/astral-sh/uv):  
   ```bash
   uv venv .venv
   uv pip install -r requirements.txt
   ```

2. Install `pre-commit` and configure hooks:
   ```bash
   uv pip install pre-commit
   pre-commit install
   ```

3. Run the pipeline:
   ```bash
   python src/run_pipeline.py
   ```

## Folder Structure

```
├── data/
│   └── raw/               # Raw dataset (CSV)
├── notebooks/             # EDA + Jupyter notebooks
├── models/                # Saved trained model
├── reports/               # Evaluation metrics
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── run_pipeline.py
```
notebooks folder contains the EDA and the Jupyter notebooks. I created Jupyter versions of the different stages of the pipeline to allow me to run the different sections easier for troubleshooting. The same functions can be found in the src folder and are later accessed in the run_pipeline.py file.

## Pre-commit Configuration  
The following pre-commit hooks were used:
- **ruff**: Ensures consistent formatting and linting.
- **nbstripout**: Strips output from Jupyter notebooks before committing.

These pre-commit hooks were selected to ensure code quality and maintain a lightweight repository by removing notebook outputs.

## Reflection
Since it's my first time using pre-commit hooks, resolving the flags by the pre-commit hooks were both easy and a bit difficult at the same time. It was difficult in a sense that these are things I do not consider when coding so it's a bit tricky to figure out what it was flagging, but they're also easy to resolve once you find them (only found out that ruff can autofix issues a lot later).