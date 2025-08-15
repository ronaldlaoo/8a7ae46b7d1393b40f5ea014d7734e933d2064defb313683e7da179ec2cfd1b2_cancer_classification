import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from pathlib import Path
from loguru import logger

def ingest_data() -> str:
    logger.info("Starting data ingestion for breast cancer dataset")

    # Find repo root (first parent with a .git)
    root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
    logger.debug(f"Detected repo root: {root}")

    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    logger.debug(f"Ensured data directory exists: {data_dir}")

    cancer = load_breast_cancer()
    logger.debug("Loaded breast cancer dataset from sklearn")

    df = pd.DataFrame(
        np.c_[cancer.data, cancer.target],
        columns=[*cancer.feature_names, "target"],
    )
    logger.debug(f"DataFrame shape: {df.shape}")

    out = data_dir / "cancer_data.csv"
    df.to_csv(out, index=False)
    logger.info(f"CSV saved to {out}")

    return str(out)