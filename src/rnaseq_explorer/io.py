"""Loading and validating count matrices and sample metadata."""
from __future__ import annotations

import pandas as pd


class InputError(ValueError):
    """Raised when uploaded data is malformed. Messages are user-facing."""


def load_counts(path_or_buffer, sep: str | None = None) -> pd.DataFrame:
    """Read a genes x samples count matrix (first column = gene IDs)."""
    df = pd.read_csv(path_or_buffer, sep=sep, engine="python", index_col=0)
    df.index = df.index.astype(str)
    if df.index.duplicated().any():
        dups = df.index[df.index.duplicated()].unique()[:5].tolist()
        raise InputError(f"Duplicate gene IDs found, e.g. {dups}. Aggregate them first.")
    return df


def load_metadata(path_or_buffer, sep: str | None = None) -> pd.DataFrame:
    """Read sample metadata (first column = sample IDs)."""
    df = pd.read_csv(path_or_buffer, sep=sep, engine="python", index_col=0)
    df.index = df.index.astype(str)
    return df


def validate(counts: pd.DataFrame, metadata: pd.DataFrame, condition_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check shapes/values, align samples, and return clean (counts, metadata).

    Raises InputError with a readable message on any problem.
    """
    if condition_col not in metadata.columns:
        raise InputError(f"Column '{condition_col}' not found in metadata. Available: {list(metadata.columns)}")

    shared = [s for s in counts.columns if s in metadata.index]
    if len(shared) < 4:
        raise InputError(
            f"Only {len(shared)} samples match between counts columns and metadata index. "
            "Sample names must be identical in both files."
        )
    dropped = set(counts.columns) ^ set(metadata.index)
    counts, metadata = counts[shared], metadata.loc[shared]

    counts = counts.apply(pd.to_numeric, errors="coerce")
    if counts.isna().any().any():
        raise InputError("Count matrix contains non-numeric or missing values.")
    if (counts < 0).any().any():
        raise InputError("Count matrix contains negative values; raw integer counts are required.")
    if ((counts % 1) != 0).any().any():
        raise InputError(
            "Count matrix contains non-integer values (TPM/FPKM/normalized?). "
            "DESeq2 requires raw integer counts, e.g. from featureCounts or HTSeq."
        )
    counts = counts.astype(int)

    groups = metadata[condition_col].astype(str).value_counts()
    if len(groups) < 2:
        raise InputError(f"'{condition_col}' has fewer than 2 levels.")
    if (groups < 2).any():
        raise InputError(f"Each group needs >= 2 replicates. Counts per level: {groups.to_dict()}")

    metadata = metadata.copy()
    metadata[condition_col] = metadata[condition_col].astype(str)
    counts.attrs["dropped_samples"] = sorted(dropped)
    return counts, metadata
