from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_DATASET_FILENAME = "NY-House-Dataset.csv"


def resolve_dataset_path() -> Path:
    """Resolve dataset path from DATASET_PATH, local data directory, or KaggleHub."""
    env_path = os.getenv("DATASET_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.exists():
            return candidate
        logger.warning(
            "DATASET_PATH does not exist (%s). Falling back to local/Kaggle sources.",
            candidate,
        )

    local_candidate = DATA_DIR / DEFAULT_DATASET_FILENAME
    if local_candidate.exists():
        return local_candidate

    try:
        import kagglehub
        downloaded_path = Path(
            kagglehub.dataset_download("nelgiriyewithana/new-york-housing-market")
        ) / DEFAULT_DATASET_FILENAME

        if not downloaded_path.exists():
            raise FileNotFoundError(f"Downloaded dataset not found at: {downloaded_path}")

        return downloaded_path
    except Exception as exc:
        raise FileNotFoundError(
            "No dataset found. Set DATASET_PATH, place a CSV in data/, or configure Kaggle credentials."
        ) from exc


def load_dataset() -> pd.DataFrame:
    """Load the default dataset (NY Housing or configured path)."""
    dataset_path = resolve_dataset_path()
    logger.info("Loading dataset from: %s", dataset_path)
    df = pd.read_csv(dataset_path)
    logger.info("Dataset loaded with shape: %s", df.shape)
    return df


def load_dataframe_from_upload(file_path: str) -> tuple[pd.DataFrame, str]:
    """Load a user-uploaded CSV or TSV file into a DataFrame.

    Args:
        file_path: Path to the uploaded file (provided by Gradio).

    Returns:
        Tuple of (dataframe, status_message).

    Raises:
        ValueError: If the file cannot be parsed as tabular data.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix in (".tsv", ".txt"):
            df = pd.read_csv(path, sep="\t")
        else:
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.read_csv(path, sep=None, engine="python")

        if df.empty or len(df.columns) < 2:
            raise ValueError("File loaded but appears empty or has only one column.")

        status = (
            f"Loaded **{path.name}** — "
            f"{df.shape[0]:,} rows x {df.shape[1]} columns. "
            f"Columns: {', '.join(str(c) for c in df.columns[:8])}"
            + (" ..." if len(df.columns) > 8 else ".")
        )
        logger.info("Uploaded dataset loaded: %s — shape %s", path.name, df.shape)
        return df, status

    except Exception as exc:
        raise ValueError(f"Could not parse '{path.name}': {exc}") from exc
