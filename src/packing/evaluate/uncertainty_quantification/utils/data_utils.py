"""Data loading and normalization utilities for uncertainty quantification."""

import pandas as pd
from typing import List, Optional


def validate_csv_columns(df: pd.DataFrame, required_cols: List[str], csv_path: str) -> None:
    """
    Validate that required columns exist in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_cols : List[str]
        List of required column names.
    csv_path : str
        Path to the CSV file (for error messages).

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV at {csv_path} is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )


def normalize_dataframe_classification(
    csv_path: str,
    text_col: str = "claim",
    label_col: str = "is_hallu",
) -> pd.DataFrame:
    """
    Load and normalize CSV data for binary classification tasks.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    text_col : str
        Name of the text column (default: "claim").
    label_col : str
        Name of the binary label column (default: "is_hallu").

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with cleaned text and numeric labels.

    Raises
    ------
    ValueError
        If required columns are missing, labels are non-numeric, or no valid rows remain.
    """
    df = pd.read_csv(csv_path)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV at {csv_path} must contain '{text_col}' and '{label_col}' columns. "
            f"Found columns: {list(df.columns)}"
        )

    df[text_col] = df[text_col].astype(str).str.strip()
    # Truncate text to 300 characters to avoid long sequences that might cause OOMs
    df[text_col] = df[text_col].str.slice(start=0, stop=300)
    df = df[df[text_col] != ""]

    # Raise an error if there are any non-numeric labels
    non_numeric_rows = ~pd.to_numeric(df[label_col], errors="coerce").notnull()
    if non_numeric_rows.any():
        raise ValueError(
            f"CSV at {csv_path} contains non-numeric values in '{label_col}' column at rows: {df[non_numeric_rows].index.tolist()}."
        )
    df[label_col] = df[label_col].astype(float)

    if len(df) == 0:
        raise ValueError(f"No valid rows left after cleaning CSV at {csv_path}")

    return df.reset_index(drop=True)


def normalize_dataframe_generation(
    csv_path: str,
    text_col: str = "model_generation",
    required_metric_cols: Optional[List[str]] = None,
    validate_metrics: bool = True,
) -> pd.DataFrame:
    """
    Load and normalize CSV data for generation tasks with quality metrics.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    text_col : str
        Name of the text column containing model generations (default: "model_generation").
    required_metric_cols : List[str], optional
        List of required metric columns (e.g., ["bleu", "rouge1"]). If None, no validation.
    validate_metrics : bool
        If True, validate that metric columns are numeric (default: True).

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with cleaned text. NaN values in metric columns are preserved.

    Raises
    ------
    ValueError
        If text column is missing, required metric columns are missing, or metrics are non-numeric.
    """
    df = pd.read_csv(csv_path)

    # Validate text column exists
    if text_col not in df.columns:
        raise ValueError(
            f"CSV at {csv_path} must contain '{text_col}' column. "
            f"Found columns: {list(df.columns)}"
        )

    # Validate required metric columns
    if required_metric_cols is not None:
        validate_csv_columns(df, required_metric_cols, csv_path)

    # Clean text column
    df[text_col] = df[text_col].astype(str).str.strip()
    # DO NOT truncate text for generation tasks (unlike classification)
    df = df[df[text_col] != ""]

    # Validate metric columns are numeric (if requested)
    if validate_metrics and required_metric_cols is not None:
        for metric_col in required_metric_cols:
            # Check that all non-NaN values are numeric
            non_null_values = df[metric_col].dropna()
            if len(non_null_values) > 0:
                non_numeric_mask = pd.to_numeric(non_null_values, errors="coerce").isna()
                if non_numeric_mask.any():
                    bad_rows = df[~df[metric_col].isna()][non_numeric_mask].index.tolist()
                    raise ValueError(
                        f"CSV at {csv_path} contains non-numeric values in '{metric_col}' "
                        f"column at rows: {bad_rows}."
                    )
            # Convert to float, preserving NaN
            df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    if len(df) == 0:
        raise ValueError(f"No valid rows left after cleaning CSV at {csv_path}")

    return df.reset_index(drop=True)
