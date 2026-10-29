"""Interactive Plotly figures used by the Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .de import classify

COLORS = {"Up": "#d62728", "Down": "#1f77b4", "NS": "#b0b0b0"}


def volcano(res: pd.DataFrame, padj_cut=0.05, lfc_cut=1.0, highlight: list[str] | None = None, label_top: int = 10) -> go.Figure:
    df = res.dropna(subset=["padj"]).copy()
    df["category"] = classify(df, padj_cut, lfc_cut)
    df["gene_id"] = df.index
    fig = px.scatter(
        df, x="log2FoldChange", y="neg_log10_padj", color="category",
        color_discrete_map=COLORS, category_orders={"category": ["Up", "Down", "NS"]},
        hover_name="gene_id",
        hover_data={"baseMean": ":.1f", "log2FoldChange": ":.2f", "padj": ":.2e", "neg_log10_padj": False, "category": False},
        opacity=0.7, render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=6))
    fig.add_hline(y=-np.log10(padj_cut), line_dash="dash", line_color="grey")
    fig.add_vline(x=lfc_cut, line_dash="dash", line_color="grey")
    fig.add_vline(x=-lfc_cut, line_dash="dash", line_color="grey")

    sig = df[df.category != "NS"].nsmallest(label_top, "padj")
    fig.add_trace(go.Scatter(
        x=sig["log2FoldChange"], y=sig["neg_log10_padj"], mode="text", text=sig["gene_id"],
        textposition="top center", textfont=dict(size=9), showlegend=False, hoverinfo="skip"))
    if highlight:
        h = df.loc[df.index.intersection(highlight)]
        fig.add_trace(go.Scatter(
            x=h["log2FoldChange"], y=h["neg_log10_padj"], mode="markers+text", text=h["gene_id"],
            textposition="bottom center",
            marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
            name="Selected", hoverinfo="skip"))
    fig.update_layout(title="Volcano plot", xaxis_title="log2 fold change", yaxis_title="-log10(adjusted p-value)",
                      template="simple_white", legend_title_text="", height=560, dragmode="select")
    return fig


def ma_plot(res: pd.DataFrame, padj_cut=0.05, lfc_cut=1.0) -> go.Figure:
    df = res.dropna(subset=["padj"]).copy()
    df["category"] = classify(df, padj_cut, lfc_cut)
    df["gene_id"] = df.index
    df["log10_baseMean"] = np.log10(df["baseMean"] + 1)
    fig = px.scatter(df, x="log10_baseMean", y="log2FoldChange", color="category", color_discrete_map=COLORS,
                     hover_name="gene_id", opacity=0.7, render_mode="webgl")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title="MA plot", xaxis_title="log10(mean normalized count + 1)", yaxis_title="log2 fold change",
                      template="simple_white", legend_title_text="", height=520)
    return fig


def pca_plot(scores: pd.DataFrame, var_ratio, metadata: pd.DataFrame, color_by: str, symbol_by: str | None = None) -> go.Figure:
    df = scores.join(metadata)
    df.index.name = None
    df["sample"] = df.index
    fig = px.scatter(df, x="PC1", y="PC2", color=color_by, symbol=symbol_by, hover_name="sample", text="sample")
    fig.update_traces(marker=dict(size=14), textposition="top center", textfont=dict(size=9))
    fig.update_layout(title="PCA of most variable genes",
                      xaxis_title=f"PC1 ({var_ratio[0]*100:.1f}% variance)",
                      yaxis_title=f"PC2 ({var_ratio[1]*100:.1f}% variance)",
                      template="simple_white", height=520)
    return fig


def heatmap(log_norm: pd.DataFrame, genes: list[str], metadata: pd.DataFrame, condition_col: str, cluster: bool = True) -> go.Figure:
    from scipy.cluster.hierarchy import leaves_list, linkage

    mat = log_norm.loc[genes]
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, 1), axis=0)
    if cluster and len(z) > 2:
        z = z.iloc[leaves_list(linkage(z.values, method="average", metric="correlation"))]
    order = metadata.sort_values(condition_col).index
    z = z[order]
    labels = [f"{s} ({metadata.loc[s, condition_col]})" for s in z.columns]
    fig = px.imshow(z.values, x=labels, y=z.index, color_continuous_scale="RdBu_r", zmin=-2.5, zmax=2.5,
                    aspect="auto", labels=dict(color="z-score"))
    fig.update_layout(title=f"Top {len(genes)} genes (row z-score of log2 normalized counts)",
                      height=max(450, 14 * len(genes) + 150), template="simple_white")
    fig.update_yaxes(tickfont=dict(size=8))
    return fig


def gene_boxplot(log_norm: pd.DataFrame, gene: str, metadata: pd.DataFrame, condition_col: str) -> go.Figure:
    df = metadata.copy()
    df.index.name = None
    df["expr"] = log_norm.loc[gene, df.index].values
    df["sample"] = df.index
    fig = px.box(df, x=condition_col, y="expr", color=condition_col, points="all", hover_name="sample")
    fig.update_layout(title=f"{gene}", yaxis_title="log2(normalized count + 1)", template="simple_white",
                      showlegend=False, height=420)
    return fig
