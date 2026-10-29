"""Static (matplotlib) versions of the figures for README / CLI use."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .de import classify

PAL = {"Up": "#d62728", "Down": "#1f77b4", "NS": "#b0b0b0"}


def volcano_png(res, path, padj_cut=0.05, lfc_cut=1.0, label_top=5):
    df = res.dropna(subset=["padj"]).copy()
    df["cat"] = classify(df, padj_cut, lfc_cut)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for c in ["NS", "Down", "Up"]:
        d = df[df.cat == c]
        ax.scatter(d.log2FoldChange, d.neg_log10_padj, s=10, c=PAL[c], alpha=0.7, label=f"{c} ({len(d)})", linewidths=0)
    ax.axhline(-np.log10(padj_cut), ls="--", c="grey", lw=0.8)
    ax.axvline(lfc_cut, ls="--", c="grey", lw=0.8); ax.axvline(-lfc_cut, ls="--", c="grey", lw=0.8)
    for g, r in df[df.cat != "NS"].nsmallest(label_top, "padj").iterrows():
        ax.annotate(g, (r.log2FoldChange, r.neg_log10_padj), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set(xlabel="log2 fold change (Treated vs Control)", ylabel="-log10 adjusted p-value", title="Volcano plot")
    ax.legend(frameon=False); sns.despine()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def pca_png(scores, var_ratio, metadata, color_by, path):
    df = scores.join(metadata)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue=color_by, style="batch" if "batch" in df else None, s=120, ax=ax)
    ax.set(xlabel=f"PC1 ({var_ratio[0]*100:.1f}%)", ylabel=f"PC2 ({var_ratio[1]*100:.1f}%)", title="PCA (top 500 variable genes)")
    sns.despine(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def heatmap_png(log_norm, genes, metadata, condition_col, path):
    mat = log_norm.loc[genes]
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, 1), axis=0)
    order = metadata.sort_values(condition_col).index
    lut = dict(zip(sorted(metadata[condition_col].unique()), ["#4c78a8", "#f58518"]))
    colors = metadata.loc[order, condition_col].map(lut)
    g = sns.clustermap(z[order], col_cluster=False, cmap="RdBu_r", vmin=-2.5, vmax=2.5, figsize=(6, 7),
                       col_colors=colors, yticklabels=False, metric="correlation", method="average",
                       cbar_kws={"label": "row z-score"})
    g.fig.suptitle(f"Top {len(genes)} DE genes", y=1.0)
    g.savefig(path, dpi=200); plt.close("all")


def ma_png(res, path, padj_cut=0.05, lfc_cut=1.0):
    df = res.dropna(subset=["padj"]).copy()
    df["cat"] = classify(df, padj_cut, lfc_cut)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in ["NS", "Down", "Up"]:
        d = df[df.cat == c]
        ax.scatter(np.log10(d.baseMean + 1), d.log2FoldChange, s=9, c=PAL[c], alpha=0.7, linewidths=0, label=c)
    ax.axhline(0, c="k", lw=0.8)
    ax.set(xlabel="log10(mean normalized count + 1)", ylabel="log2 fold change", title="MA plot")
    ax.legend(frameon=False); sns.despine(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
