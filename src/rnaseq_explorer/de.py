"""Differential expression with PyDESeq2 plus helpers for QC."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats


@dataclass
class DEResult:
    """Container for everything the plots need."""

    results: pd.DataFrame        # per-gene stats (log2FC, padj, ...)
    log_norm: pd.DataFrame       # log2(normalized counts + 1), genes x samples
    norm_counts: pd.DataFrame    # median-of-ratios normalized counts
    metadata: pd.DataFrame
    condition_col: str
    reference: str
    test: str
    n_genes_input: int
    n_genes_tested: int


def run_deseq2(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    condition_col: str,
    reference: str,
    test: str,
    covariates: list[str] | None = None,
    min_total_count: int = 10,
    lfc_shrink: bool = False,
    n_cpus: int = 1,
) -> DEResult:
    """Run the PyDESeq2 pipeline: size factors -> dispersions -> Wald test.

    Parameters
    ----------
    counts : genes x samples integer matrix.
    metadata : samples x variables, index must match ``counts.columns``.
    reference, test : levels of ``condition_col``; log2FC is ``test`` vs ``reference``.
    covariates : extra columns (e.g. ["batch"]) added to the design formula.
    min_total_count : genes with fewer summed reads are removed before fitting.
    lfc_shrink : apply apeGLM shrinkage to the reported log2FC.
    """
    covariates = covariates or []
    keep = counts.sum(axis=1) >= min_total_count
    filtered = counts.loc[keep]
    if filtered.shape[0] < 10:
        raise ValueError("Fewer than 10 genes remain after filtering; lower the minimum count.")

    sub_meta = metadata[metadata[condition_col].isin([reference, test])]
    sub_counts = filtered[sub_meta.index]
    design_cols = [condition_col] + covariates
    design = "~" + " + ".join(design_cols)

    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=sub_counts.T,  # PyDESeq2 expects samples x genes
        metadata=sub_meta[design_cols],
        design=design,
        refit_cooks=True,
        inference=inference,
        quiet=True,
    )
    dds.deseq2()

    stats = DeseqStats(dds, contrast=[condition_col, test, reference], inference=inference, quiet=True)
    stats.summary()
    if lfc_shrink:
        stats.lfc_shrink(coeff=f"{condition_col}[T.{test}]")
    res = stats.results_df.copy()
    res.index.name = "gene_id"
    res["neg_log10_padj"] = -np.log10(res["padj"].clip(lower=1e-300))
    res["neg_log10_pvalue"] = -np.log10(res["pvalue"].clip(lower=1e-300))

    norm = pd.DataFrame(dds.layers["normed_counts"], index=dds.obs_names, columns=dds.var_names).T
    return DEResult(
        results=res,
        log_norm=np.log2(norm + 1),
        norm_counts=norm,
        metadata=sub_meta,
        condition_col=condition_col,
        reference=reference,
        test=test,
        n_genes_input=counts.shape[0],
        n_genes_tested=filtered.shape[0],
    )


def classify(res: pd.DataFrame, padj_cut: float = 0.05, lfc_cut: float = 1.0) -> pd.Series:
    """Label each gene Up / Down / NS given thresholds."""
    sig = (res["padj"] < padj_cut) & (res["log2FoldChange"].abs() >= lfc_cut)
    label = pd.Series("NS", index=res.index)
    label[sig & (res["log2FoldChange"] > 0)] = "Up"
    label[sig & (res["log2FoldChange"] < 0)] = "Down"
    return label


def pca(log_norm: pd.DataFrame, n_top: int = 500, n_components: int = 2):
    """PCA on the ``n_top`` most variable genes. Returns (scores_df, explained_variance_ratio)."""
    from sklearn.decomposition import PCA

    top = log_norm.var(axis=1).nlargest(min(n_top, len(log_norm))).index
    X = log_norm.loc[top].T
    X = X - X.mean(axis=0)
    p = PCA(n_components=n_components, random_state=0)
    scores = pd.DataFrame(
        p.fit_transform(X), index=X.index, columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return scores, p.explained_variance_ratio_
