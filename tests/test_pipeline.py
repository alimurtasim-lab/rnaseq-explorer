from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rnaseq_explorer import de, io

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def loaded():
    c, m = io.load_counts(DATA / "demo_counts.csv"), io.load_metadata(DATA / "demo_metadata.csv")
    return io.validate(c, m, "condition")


@pytest.fixture(scope="module")
def result(loaded):
    c, m = loaded
    return de.run_deseq2(c, m, "condition", "Control", "Treated", covariates=["batch"])


def test_recovers_ground_truth(result):
    truth = pd.read_csv(DATA / "demo_ground_truth.csv", index_col=0)
    res = result.results
    called = de.classify(res) != "NS"
    j = truth.join(called.rename("called"), how="inner")
    precision = (j.called & j.is_de).sum() / max(j.called.sum(), 1)
    recall = (j.called & j.is_de).sum() / j.is_de.sum()
    assert precision > 0.85 and recall > 0.75


def test_fold_change_direction(result):
    truth = pd.read_csv(DATA / "demo_ground_truth.csv", index_col=0)
    j = result.results.join(truth, how="inner")
    strong = j[(j.padj < 0.01) & j.is_de]
    assert (np.sign(strong.log2FoldChange) == np.sign(strong.true_log2FC)).all()


def test_rejects_non_integer_counts(loaded):
    c, m = loaded
    with pytest.raises(io.InputError, match="non-integer"):
        io.validate(c.astype(float) + 0.5, m, "condition")


def test_rejects_missing_column(loaded):
    c, m = loaded
    with pytest.raises(io.InputError, match="not found"):
        io.validate(c, m, "nonexistent")


def test_pca_shape(result):
    scores, var = de.pca(result.log_norm, 200)
    assert scores.shape == (12, 2) and 0 < var.sum() <= 1


def test_streamlit_app_runs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 7
